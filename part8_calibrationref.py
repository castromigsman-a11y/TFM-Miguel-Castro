#!/usr/bin/env python3
# part8 - calibration curves
# makes calibration curves (plus the integrated calibration index, ici) for the
# cif at 5 and 10 years, for both fine-gray and deephit, on the two competing
# outcomes (cv death = 1, cancer death = 2).
#
# why this exists: the thesis already reports the brier score, but that mixes
# discrimination and calibration together. the supervisor asked for actual
# calibration graphs. so here the "observed" risk is estimated the
# competing-risks-correct way (aalen-johansen within bins of predicted risk),
# with a smoothed curve and the ici on top.
#
# inputs (all already sitting on disk from the earlier parts):
#   nhanes_imputed_stacked.csv      - the imputed data (.imp, SEQN, event, time, covars)
#   finegray_test_predictions.csv   - per-subject fine-gray cifs + the fold column
#   deephit_best_params.json        - the fixed deephit hyperparameters
#   (don't need cv_fold_indices.csv here, the folds come from the fg predictions)
#
# outputs:
#   calibration_cv_5yr.png, calibration_cv_10yr.png,
#   calibration_cancer_5yr.png, calibration_cancer_10yr.png
#   calibration_combined.png        - all four in a 2x2 panel
#   calibration_ici_summary.csv     - ici + e50/e90 per model/cause/horizon
#   calibration_console_summary.txt
#
# it refits deephit on imputation 1 with the same folds as fine-gray, so both
# models' predictions are out-of-fold and directly comparable. ~5-10 min.

import numpy as np, pandas as pd, json, warnings, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch, torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lifelines import AalenJohansenFitter
from statsmodels.nonparametric.smoothers_lowess import lowess

SEED = 2024
NUM_DURATIONS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
horizons = {"5yr": 60, "10yr": 120}        # months
causes  = {1: "Cardiovascular death", 2: "Cancer death"}
np.random.seed(SEED); torch.manual_seed(SEED)

feature_cols = ['age','sex','poverty_ratio','bmi','systolic_bp','diastolic_bp',
    'hypertension','diabetes','chd','stroke','alcohol_past_year','hba1c',
    'total_cholesterol','hdl_cholesterol','serum_creatinine','fasting_glucose',
    'race_ethnicity_enc','education_enc','smoking_status_enc']

print("="*68); print("part8: CIF calibration curves + ICI (Fine-Gray vs DeepHit)")
print("Device:", DEVICE); print("="*68)

# load
df_all = pd.read_csv("nhanes_imputed_stacked.csv")
fg     = pd.read_csv("finegray_test_predictions.csv")     # has the fold column
with open("deephit_best_params.json") as f: bp = json.load(f)
folds = fg[["SEQN","fold"]].drop_duplicates()

# --- observed cif ----------------------------------------------------------
# aalen-johansen observed cif for one cause at the horizon, among the people in
# the given mask (i.e. within one bin of predicted risk)
def observed_cif_in_mask(durations, events, cause, horizon, mask):
    d = np.asarray(durations)[mask]; e = np.asarray(events)[mask]
    if len(d) < 5 or (e == cause).sum() == 0:
        return np.nan
    ajf = AalenJohansenFitter(calculate_variance=False)
    try:
        ajf.fit(d, e, event_of_interest=cause)
        cif = ajf.cumulative_density_
        # the cif value at the horizon = the last estimate at a time <= horizon
        idx = cif.index[cif.index <= horizon]
        if len(idx) == 0: return 0.0
        return float(cif.loc[idx.max()].values[0])
    except Exception:
        return np.nan

# split people into bins by predicted risk, then for each bin compare the mean
# prediction against the observed (aalen-johansen) rate
def calibration_points(pred, durations, events, cause, horizon, n_bins=10):
    pred = np.asarray(pred)
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    mp, ob, ct = [], [], []
    for b in bins:
        if len(b) == 0: continue
        m = np.zeros(len(pred), dtype=bool); m[b] = True
        mp.append(pred[b].mean())
        ob.append(observed_cif_in_mask(durations, events, cause, horizon, m))
        ct.append(len(b))
    return np.array(mp), np.array(ob), np.array(ct)

# integrated calibration index: lowess-smooth the observed event indicator
# against the prediction, then average the absolute gap between the smoothed
# curve and the diagonal. people censored before the horizon get dropped, which
# is the standard (slightly conservative) choice.
def ici(pred, durations, events, cause, horizon):
    pred = np.asarray(pred); d = np.asarray(durations); e = np.asarray(events)
    obs_event = ((d <= horizon) & (e == cause)).astype(float)
    keep = ~((d < horizon) & (e == 0))          # drop censored-before-horizon
    p, y = pred[keep], obs_event[keep]
    if len(p) < 20: return np.nan, np.nan, np.nan
    sm = lowess(y, p, frac=0.6, return_sorted=True)
    cal = np.interp(p, sm[:,0], sm[:,1])
    diff = np.abs(cal - p)
    # ici is the mean gap, e50 and e90 are the median and 90th percentile gaps
    return float(np.mean(diff)), float(np.percentile(diff,50)), float(np.percentile(diff,90))

# --- refit deephit out-of-fold (imputation 1, same folds as fine-gray) -----
def deephit_oof_predictions():
    df = df_all[df_all['.imp']==1].copy().reset_index(drop=True)
    if 'followup_months' not in df or df['followup_months'].isna().all():
        df['followup_months'] = df['followup_years']*12
    for c in ['race_ethnicity','education','smoking_status']:
        df[c+'_enc'] = LabelEncoder().fit_transform(df[c].astype(str).fillna('Unknown'))
    df = df.merge(folds, on='SEQN', how='left')
    lab = LabTransDiscreteTime(NUM_DURATIONS)
    lab.fit(df['followup_months'].values.astype('float32'), df['event'].values.astype('int64'))
    cuts = lab.cuts; n_dur=len(cuts)
    class Reshape(torch.nn.Module):
        def __init__(s,nr,nd): super().__init__(); s.nr,s.nd=nr,nd
        def forward(s,x): return x.view(-1,s.nr,s.nd)
    def net():
        L=[]; prev=len(feature_cols)
        for _ in range(bp['num_layers']):
            L+=[torch.nn.Linear(prev,bp['num_nodes']),torch.nn.BatchNorm1d(bp['num_nodes']),
                torch.nn.ReLU(),torch.nn.Dropout(bp['dropout'])]; prev=bp['num_nodes']
        L+=[torch.nn.Linear(prev,3*n_dur),Reshape(3,n_dur)]
        return torch.nn.Sequential(*L)
    def lab_tr(dur,ev):
        idx=np.clip(np.searchsorted(cuts,dur,side='right')-1,0,n_dur-1)
        return idx.astype('int64'), ev.astype('int64')
    out = pd.DataFrame({'SEQN':df['SEQN']})
    for hl in horizons: out[f'dh_cv_{hl}']=np.nan; out[f'dh_cancer_{hl}']=np.nan
    # train on the other 4 folds, predict the held-out one, loop over all folds
    for fold in sorted(df['fold'].dropna().unique()):
        tr=df['fold']!=fold; te=df['fold']==fold
        sc=StandardScaler()
        Xtr=sc.fit_transform(df.loc[tr,feature_cols].astype('float32')).astype('float32')
        Xte=sc.transform(df.loc[te,feature_cols].astype('float32')).astype('float32')
        ytr=lab_tr(df.loc[tr,'followup_months'].values.astype('float32'),
                   df.loc[tr,'event'].values.astype('int64'))
        n=len(Xtr); nv=int(.15*n); pm=np.random.RandomState(SEED+int(fold)).permutation(n)
        Xt,Xv=Xtr[pm[nv:]],Xtr[pm[:nv]]
        yt=(ytr[0][pm[nv:]],ytr[1][pm[nv:]]); yv=(ytr[0][pm[:nv]],ytr[1][pm[:nv]])
        m=DeepHit(net(),tt.optim.Adam,alpha=bp['alpha'],sigma=bp['sigma'],duration_index=cuts)
        m.optimizer.set_lr(bp['lr'])
        m.fit(Xt,yt,batch_size=bp['batch_size'],epochs=300,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)],val_data=(Xv,yv),verbose=False)
        cif=m.predict_cif(Xte)            # [risk, time, subj]
        teidx=df.index[te]
        for hl,hm in horizons.items():
            ti=min(np.searchsorted(cuts,hm,side='right')-1,n_dur-1)
            out.loc[teidx,f'dh_cv_{hl}']     = cif[0][ti,:]
            out.loc[teidx,f'dh_cancer_{hl}'] = cif[1][ti,:]
        print(f"  DeepHit fold {int(fold)} done")
    return df[['SEQN','followup_months','event']].merge(out,on='SEQN')

print("\nRefitting DeepHit out-of-fold (imputation 1)...")
dh = deephit_oof_predictions()
base = dh.merge(fg, on='SEQN', how='inner')

# --- plots + ici -----------------------------------------------------------
plt.rcParams.update({'font.size':10,'figure.dpi':150})
ici_rows=[]; panels=[]
for cause, clab in causes.items():
    cname = 'cv' if cause==1 else 'cancer'
    for hl, hm in horizons.items():
        fg_col=f'cif_{cname}_{hl}'; dh_col=f'dh_{cname}_{hl}'
        dur=base['followup_months'].values; ev=base['event'].values
        fig,ax=plt.subplots(figsize=(5.2,5.0))
        # work out both models' calibration points first so the axes can be
        # zoomed to the data actually plotted, not stretched out to some lone
        # outlier prediction. keeping x and y on the same scale so the 45-degree
        # line stays a fair reference.
        series=[]; panel_max=0.0
        for col,label,color in [(fg_col,'Fine-Gray','#2c7fb8'),(dh_col,'DeepHit','#de2d26')]:
            pred=base[col].values
            mp,ob,ct=calibration_points(pred,dur,ev,cause,hm,n_bins=10)
            ok=~np.isnan(ob)
            series.append((mp[ok],ob[ok],label,color))
            if ok.any():
                panel_max=max(panel_max, np.nanmax(mp[ok]), np.nanmax(ob[ok]))
            i,e50,e90=ici(pred,dur,ev,cause,hm)
            ici_rows.append(dict(model=label,cause=clab,horizon=hl,ICI=i,E50=e50,E90=e90))
        mx=max(0.05, panel_max*1.15)          # zoom to the plotted points, with a small floor
        ax.plot([0,mx],[0,mx],'--',color='gray',lw=1,label='Ideal')
        for mpx,oby,label,color in series:
            ax.plot(mpx,oby,'o-',color=color,ms=5,lw=1.5,label=label)
        ax.set_xlim(0,mx); ax.set_ylim(0,mx)
        ax.set_xlabel('Predicted CIF'); ax.set_ylabel('Observed CIF (Aalen-Johansen)')
        ax.set_title(f'{clab} \u2014 {hl.replace("yr"," year")}')
        ax.legend(loc='upper left',frameon=False,fontsize=9)
        ax.set_aspect('equal','box'); fig.tight_layout()
        fn=f'calibration_{cname}_{hl}.png'; fig.savefig(fn,bbox_inches='tight'); plt.close(fig)
        panels.append(fn); print("  wrote",fn)

# stitch the four panels into one 2x2 figure
fig,axes=plt.subplots(2,2,figsize=(10,10))
for axp,fn in zip(axes.ravel(),panels):
    axp.imshow(plt.imread(fn)); axp.axis('off')
fig.tight_layout(); fig.savefig('calibration_combined.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print("  wrote calibration_combined.png")

ici_df=pd.DataFrame(ici_rows); ici_df.to_csv('calibration_ici_summary.csv',index=False)
lines=["CALIBRATION SUMMARY (Integrated Calibration Index)","="*52,
       "ICI = mean |predicted - observed|; lower is better.",""]
for (c,h),g in ici_df.groupby(['cause','horizon']):
    lines.append(f"{c}, {h}:")
    for _,r in g.iterrows():
        lines.append(f"   {r['model']:10s} ICI={r['ICI']:.4f}  E50={r['E50']:.4f}  E90={r['E90']:.4f}")
    lines.append("")
open('calibration_console_summary.txt','w').write("\n".join(lines))
print("\n".join(lines))
print("DONE. Send back calibration_combined.png and calibration_ici_summary.csv.")
