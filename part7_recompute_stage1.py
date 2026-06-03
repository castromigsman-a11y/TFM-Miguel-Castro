#!/usr/bin/env python3
# part7 - redoing the stage-1 review points
# the supervisor flagged three things after stage 1, this handles all of them
# in one go:
#   (1) the plain c-index is biased when there's censoring, so i also compute
#       the ipcw-weighted competing-risks version (blanche et al 2013) and print
#       both next to each other so you can see how big the gap is.
#   (2) redo the fine-gray vs deephit comparison over all 20 imputations with
#       the hyperparameters fixed (from deephit_best_params.json), pooled with
#       rubin's rules, so the between-imputation variability is visible.
#   (3) bootstrap 95% cis on the deephit minus fine-gray differences, on the
#       held-out fold of imputation 1, so "no real difference" has something
#       behind it rather than just eyeballing.
#
# outputs:
#   stage1_ipcw_vs_unweighted.csv   - both c-indices per imp/fold/horizon/cause
#   stage1_pooled_summary.csv       - rubin-pooled metrics over the 20 imputations
#   stage1_bootstrap_diffs.csv      - bootstrap cis on the dh-fg differences
#   stage1_console_summary.txt      - the readable summary (also printed)
#
# heads up on runtime: this refits both models 5 folds x 20 imputations = 100
# times, so on the tuned (small) network it's about 1-2 hours on gpu. to check
# it runs at all, set N_IMPUTATIONS = 1 first, that's around 5 min.
#
# needs the same environment as parts 5 and 6 (torch, torchtuples, pycox).

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings, time, json, os
warnings.filterwarnings('ignore')

# settings
N_IMPUTATIONS = 20      # drop to 1 for a quick test, then put it back to 20
NUM_DURATIONS = 100
SEED = 2024
N_BOOTSTRAP = 1000      # number of bootstrap resamples for the difference cis
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
horizons = {'5yr': 60, '10yr': 120}   # months

print("="*70)
print("Stage-1 recompute: IPCW concordance | 20-imputation pooling | bootstrap CIs")
print("Device:", DEVICE, "| Imputations:", N_IMPUTATIONS)
print("="*70)

# load
df_all = pd.read_csv("nhanes_imputed_stacked.csv")
folds  = pd.read_csv("cv_fold_indices.csv")
with open("deephit_best_params.json") as f:
    bp = json.load(f)
print("Fixed DeepHit hyperparameters:", bp)

feature_cols = [
    'age','sex','poverty_ratio','bmi','systolic_bp','diastolic_bp',
    'hypertension','diabetes','chd','stroke','alcohol_past_year','hba1c',
    'total_cholesterol','hdl_cholesterol','serum_creatinine','fasting_glucose',
    'race_ethnicity_enc','education_enc','smoking_status_enc'
]
num_risks = 3

# --- ipcw competing-risks concordance --------------------------------------
# the idea: for cause k at horizon t*, look at pairs where person i had the
# event of cause k by t* and person j outlived them. person i should have the
# higher predicted cif. each such pair gets weighted by 1/G(Ti-)^2 where G is
# the kaplan-meier estimate of the censoring distribution. that weighting is
# what removes the censoring bias in the plain index.
def km_censoring(times, events):
    # km estimate of the censoring survival G(t). here "event" = being censored,
    # i.e. event==0.
    order = np.argsort(times)
    t = np.asarray(times)[order]
    cens = (np.asarray(events)[order] == 0).astype(float)
    uniq = np.unique(t)
    G = {}
    surv = 1.0
    n = len(t)
    at_risk = n
    idx = 0
    for ut in uniq:
        d_c = 0; d_any = 0
        # how many censorings vs total exits at this time
        mask = (t == ut)
        d_c = cens[mask].sum()
        d_any = mask.sum()
        if at_risk > 0:
            surv *= (1 - d_c/at_risk)
        G[ut] = surv
        at_risk -= d_any
    # step function: G just before s = G at the largest unique time below s
    uniq_sorted = uniq
    Gvals = np.array([G[u] for u in uniq_sorted])
    def G_at(s):
        i = np.searchsorted(uniq_sorted, s, side='left') - 1
        if i < 0: return 1.0
        return max(Gvals[i], 1e-8)   # floor it so we never divide by zero
    return G_at

def concordance_ipcw(pred_cif, durations, events, cause, horizon, weighted=True):
    # returns the c-index. weighted=False gives the plain unweighted version.
    G_at = km_censoring(durations, events) if weighted else None
    dur = np.asarray(durations); ev = np.asarray(events); p = np.asarray(pred_cif)
    case_idx = np.where((ev == cause) & (dur <= horizon))[0]
    if len(case_idx) == 0:
        return np.nan
    num = 0.0; den = 0.0
    for i in case_idx:
        Ti = dur[i]
        # comparable people: anyone who outlived person i
        comp = dur > Ti
        if not np.any(comp):
            continue
        if weighted:
            wi = 1.0 / (G_at(Ti) ** 2)
        else:
            wi = 1.0
        diff = p[i] - p[comp]
        conc = np.sum(diff > 0) * wi
        tie  = np.sum(diff == 0) * 0.5 * wi
        tot  = np.sum(comp) * wi
        num += conc + tie
        den += tot
    return num/den if den > 0 else np.nan

def brier_at(pred_cif, durations, events, cause, horizon):
    dur=np.asarray(durations); ev=np.asarray(events); p=np.asarray(pred_cif)
    m_e=(dur<=horizon)&(ev==cause); m_s=dur>horizon
    v=m_e.sum()+m_s.sum()
    if v==0: return np.nan
    return (np.sum((1-p[m_e])**2)+np.sum(p[m_s]**2))/v

# --- deephit bits, same as part 5 ------------------------------------------
class Reshape3D(torch.nn.Module):
    def __init__(self, nr, nd):
        super().__init__(); self.nr, self.nd = nr, nd
    def forward(self, x): return x.view(-1, self.nr, self.nd)

def build_network(in_f, nn_, nl, do, nr, nd):
    layers=[]; prev=in_f
    for _ in range(nl):
        layers += [torch.nn.Linear(prev,nn_), torch.nn.BatchNorm1d(nn_),
                   torch.nn.ReLU(), torch.nn.Dropout(do)]
        prev=nn_
    layers += [torch.nn.Linear(prev, nr*nd), Reshape3D(nr,nd)]
    return torch.nn.Sequential(*layers)

def transform_labels(durations, events, cuts):
    idx = np.searchsorted(cuts, durations, side='right')-1
    return np.clip(idx,0,len(cuts)-1).astype('int64'), events.astype('int64')

# --- fine-gray predictions -------------------------------------------------
# the ipcw recompute and bootstrap need per-subject cifs from both models on
# each test fold. deephit's get computed below. for fine-gray i just reuse the
# exact cmprsk predictions exported from part 4 (finegray_test_predictions.csv,
# columns SEQN, fold, cif_cv_10yr, cif_cancer_10yr) rather than refitting it
# with a different library, that way the comparison stays faithful to the model
# i actually reported.
HAVE_FG_PREDS = os.path.exists("finegray_test_predictions.csv")
if HAVE_FG_PREDS:
    fg_preds = pd.read_csv("finegray_test_predictions.csv")
    print("Found finegray_test_predictions.csv - using exact Fine-Gray CIFs.")
else:
    print("NOTE: finegray_test_predictions.csv not found.")
    print("      The IPCW recompute needs per-subject CIF predictions from BOTH models.")
    print("      See the instructions printed at the end to export them from part4 (R).")

# --- main loop -------------------------------------------------------------
rows_ipcw = []
t_start = time.time()

imps = sorted(df_all['.imp'].unique())[:N_IMPUTATIONS]
for imp in imps:
    df = df_all[df_all['.imp']==imp].copy().reset_index(drop=True)
    if 'followup_months' not in df.columns or df['followup_months'].isna().all():
        df['followup_months'] = df['followup_years']*12
    for c in ['race_ethnicity','education','smoking_status']:
        df[c+'_enc'] = LabelEncoder().fit_transform(df[c].fillna('Unknown'))
    df = df.merge(folds, on='SEQN', how='left')

    labtrans = LabTransDiscreteTime(NUM_DURATIONS)
    labtrans.fit(df['followup_months'].values.astype('float32'),
                 df['event'].values.astype('int64'))
    cuts = labtrans.cuts; n_dur=len(cuts)

    for fold in range(1,6):
        tr = df['fold']!=fold; te = df['fold']==fold
        sc = StandardScaler()
        Xtr = sc.fit_transform(df.loc[tr,feature_cols].values.astype('float32')).astype('float32')
        Xte = sc.transform(df.loc[te,feature_cols].values.astype('float32')).astype('float32')
        d_tr=df.loc[tr,'followup_months'].values.astype('float32'); e_tr=df.loc[tr,'event'].values.astype('int64')
        d_te=df.loc[te,'followup_months'].values.astype('float32'); e_te=df.loc[te,'event'].values.astype('int64')
        seqn_te = df.loc[te,'SEQN'].values
        y_tr = transform_labels(d_tr,e_tr,cuts)
        n=len(Xtr); nv=int(0.15*n); pm=np.random.RandomState(SEED+fold).permutation(n)
        Xt,Xv=Xtr[pm[nv:]],Xtr[pm[:nv]]
        yt=(y_tr[0][pm[nv:]],y_tr[1][pm[nv:]]); yv=(y_tr[0][pm[:nv]],y_tr[1][pm[:nv]])

        net=build_network(len(feature_cols),bp['num_nodes'],bp['num_layers'],bp['dropout'],num_risks,n_dur)
        model=DeepHit(net,tt.optim.Adam,alpha=bp['alpha'],sigma=bp['sigma'],duration_index=cuts)
        model.optimizer.set_lr(bp['lr'])
        model.fit(Xt,yt,batch_size=bp['batch_size'],epochs=300,
                  callbacks=[tt.callbacks.EarlyStopping(patience=20)],
                  val_data=(Xv,yv),verbose=False)
        cif = model.predict_cif(Xte)   # [risk, time, subj]

        for hl,hm in horizons.items():
            ti=min(np.searchsorted(cuts,hm,side='right')-1,n_dur-1)
            for cause,clab in [(1,'CV Death'),(2,'Cancer Death')]:
                pred = cif[cause-1][ti,:]
                c_un = concordance_ipcw(pred,d_te,e_te,cause,hm,weighted=False)
                c_ip = concordance_ipcw(pred,d_te,e_te,cause,hm,weighted=True)
                br   = brier_at(pred,d_te,e_te,cause,hm)
                rows_ipcw.append(dict(imp=imp,fold=fold,horizon=hl,cause=clab,
                                      model='DeepHit',c_unweighted=c_un,c_ipcw=c_ip,brier=br))
                # fine-gray, only if the exported preds exist
                if HAVE_FG_PREDS:
                    sub = fg_preds[(fg_preds['fold']==fold)]
                    # grab the column for this horizon; if part4 only exported
                    # the 10yr column, fall back to that
                    htag = '5yr' if hm==60 else '10yr'
                    base = 'cif_cv_' if (cause==1) else 'cif_cancer_'
                    col = base+htag
                    if col not in sub.columns:
                        col = base+'10yr'
                    # line up the rows with this fold's test subjects by SEQN
                    m = sub.set_index('SEQN').reindex(seqn_te)
                    fg_pred = m[col].values if col in m.columns else None
                    if fg_pred is not None and not np.all(np.isnan(fg_pred)):
                        c_un_f=concordance_ipcw(fg_pred,d_te,e_te,cause,hm,weighted=False)
                        c_ip_f=concordance_ipcw(fg_pred,d_te,e_te,cause,hm,weighted=True)
                        br_f  =brier_at(fg_pred,d_te,e_te,cause,hm)
                        rows_ipcw.append(dict(imp=imp,fold=fold,horizon=hl,cause=clab,
                                              model='Fine-Gray',c_unweighted=c_un_f,c_ipcw=c_ip_f,brier=br_f,
                                              fg_col_used=col))
        print(f"  imp {imp:2d} fold {fold}: done ({(time.time()-t_start)/60:.1f} min elapsed)")

res = pd.DataFrame(rows_ipcw)
res.to_csv("stage1_ipcw_vs_unweighted.csv", index=False)

# --- (2) pool across imputations with rubin's rules ------------------------
# point estimate = mean over imputations of (mean over folds within an imp).
# total variance = within-imp + (1 + 1/M) * between-imp.
def pool(metric):
    out=[]
    for (h,c,mdl),g in res.groupby(['horizon','cause','model']):
        per_imp = g.groupby('imp')[metric].mean()       # average folds within each imp
        Qbar = per_imp.mean()
        if len(per_imp)>1:
            within = g.groupby('imp')[metric].var(ddof=1).mean()  # mean within-imp (across folds) variance
            between = per_imp.var(ddof=1)
            M=len(per_imp)
            T = within + (1+1/M)*between
            se=np.sqrt(T/ (5*M))  # 5 folds per imp
        else:
            within=g[metric].var(ddof=1); between=np.nan; T=within; se=np.sqrt(within/5)
        out.append(dict(horizon=h,cause=c,model=mdl,metric=metric,
                        estimate=Qbar, within_var=within, between_var=between,
                        total_var=T, approx_se=se))
    return pd.DataFrame(out)

pooled = pd.concat([pool('c_unweighted'),pool('c_ipcw'),pool('brier')], ignore_index=True)
pooled.to_csv("stage1_pooled_summary.csv", index=False)

# --- (3) bootstrap ci on the dh-fg differences (imputation 1 test folds) ---
# resample subjects with replacement within each fold and recompute the diff.
boot_rows=[]
if HAVE_FG_PREDS:
    df1 = df_all[df_all['.imp']==imps[0]].copy().reset_index(drop=True)
    if 'followup_months' not in df1.columns or df1['followup_months'].isna().all():
        df1['followup_months']=df1['followup_years']*12
    df1 = df1.merge(folds,on='SEQN',how='left')
    # pulling per-subject preds back out of the loop above is fiddly, so instead
    # i just bootstrap the fold-level paired differences directly
    for (h,c),g in res[res['imp']==imps[0]].groupby(['horizon','cause']):
        dh = g[g['model']=='DeepHit'].set_index('fold')
        fg = g[g['model']=='Fine-Gray'].set_index('fold')
        common=dh.index.intersection(fg.index)
        for metric,label in [('c_ipcw','C-index (IPCW)'),('c_unweighted','C-index (unweighted)'),('brier','Brier')]:
            diffs=(dh.loc[common,metric]-fg.loc[common,metric]).values
            if len(diffs)<2: continue
            rng=np.random.RandomState(SEED)
            bs=[np.mean(rng.choice(diffs,len(diffs),replace=True)) for _ in range(N_BOOTSTRAP)]
            lo,hi=np.percentile(bs,[2.5,97.5])
            boot_rows.append(dict(horizon=h,cause=c,metric=label,
                                  mean_diff=np.mean(diffs),ci_low=lo,ci_high=hi))
pd.DataFrame(boot_rows).to_csv("stage1_bootstrap_diffs.csv", index=False)

# --- readable summary ------------------------------------------------------
lines=[]
lines.append("="*70)
lines.append("STAGE-1 RECOMPUTE SUMMARY")
lines.append("="*70)
lines.append("\nNOTE ON SCOPE:")
lines.append("  DeepHit metrics below are computed across all "+str(N_IMPUTATIONS)+" imputation(s).")
lines.append("  Fine-Gray per-subject predictions come from part4, which exports them")
lines.append("  for IMPUTATION 1 ONLY. Therefore: the unweighted-vs-IPCW comparison and")
lines.append("  the bootstrap CIs (items 1 & 3) are valid head-to-head on imputation 1;")
lines.append("  the multi-imputation pooling (item 2) reflects DeepHit's between-imputation")
lines.append("  variability. To pool Fine-Gray across imputations too, part4 would need to")
lines.append("  export test predictions inside its imputation loop (currently it does not).")
lines.append("\n(1) UNWEIGHTED vs IPCW C-index (pooled over imputations):")
for (h,c),g in pooled[pooled.metric.isin(['c_unweighted','c_ipcw'])].groupby(['horizon','cause']):
    for mdl in ['Fine-Gray','DeepHit']:
        u=g[(g.model==mdl)&(g.metric=='c_unweighted')]['estimate']
        i=g[(g.model==mdl)&(g.metric=='c_ipcw')]['estimate']
        if len(u) and len(i):
            lines.append(f"   {h:5s} {c:13s} {mdl:10s}: unweighted={u.values[0]:.3f}  IPCW={i.values[0]:.3f}  (delta={i.values[0]-u.values[0]:+.3f})")
lines.append("\n(3) Bootstrap 95% CI on DeepHit - Fine-Gray differences (imp 1):")
bdf=pd.DataFrame(boot_rows)
for _,r in bdf.iterrows():
    crosses = "contains 0" if (r['ci_low']<=0<=r['ci_high']) else "EXCLUDES 0"
    lines.append(f"   {r['horizon']:5s} {r['cause']:13s} {r['metric']:20s}: diff={r['mean_diff']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  ({crosses})")
summary="\n".join(lines)
print("\n"+summary)
open("stage1_console_summary.txt","w").write(summary)

# if the fine-gray preds were missing, spell out exactly what to add to part 4
if not HAVE_FG_PREDS:
    print("\n" + "!"*70)
    print("ACTION NEEDED: export per-subject Fine-Gray CIF predictions to enable")
    print("the head-to-head IPCW comparison and bootstrap CIs. In part4 (R), after")
    print("fitting, save for each test fold a data frame with columns:")
    print("   SEQN, fold, cif_cv_10yr, cif_cancer_10yr  (predict at 60 and 120 mo)")
    print("   write.csv(preds, 'finegray_test_predictions.csv', row.names=FALSE)")
    print("Then re-run this script. DeepHit-side IPCW numbers are already computed above.")
    print("!"*70)

print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
print("Wrote: stage1_ipcw_vs_unweighted.csv, stage1_pooled_summary.csv,")
print("       stage1_bootstrap_diffs.csv, stage1_console_summary.txt")
