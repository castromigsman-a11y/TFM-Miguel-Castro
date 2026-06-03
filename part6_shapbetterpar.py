# part 6 - shap interpretability
# trains deephit on folds 2-5, computes shap values on fold 1, and produces the
# importance rankings, beeswarm plots, waterfall plots, and a spearman
# correlation between the shap ranks and the fine-gray hazard ratio ranks.
# this is by far the slowest script, see the note on the sample sizes below.

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
import shap
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, time, json

warnings.filterwarnings('ignore')

# the kernelexplainer runtime roughly scales with N_EXPLAIN x N_SHAP_SAMPLES x
# N_BACKGROUND. if it's too slow, drop N_EXPLAIN first then N_SHAP_SAMPLES.
# i pushed these up from my first pass to get a more stable importance ranking,
# it's worth the wait.
N_BACKGROUND   = 200
N_EXPLAIN      = 8000     # more instances -> steadier global ranking
N_SHAP_SAMPLES = 2000     # more coalitions -> more accurate per-person shapley values
NUM_DURATIONS  = 100
SEED = 2024

# if part 5 didn't leave a params file behind, use these instead
FALLBACK_PARAMS = {
    'num_layers': 3, 'num_nodes': 64, 'dropout': 0.4,
    'lr': 0.00112, 'batch_size': 512, 'alpha': 0.2, 'sigma': 0.2
}

print("=" * 60)
print("Part 6: SHAP Interpretability")
print("Background:", N_BACKGROUND, "| Explain:", N_EXPLAIN, "| nsamples:", N_SHAP_SAMPLES)
print("=" * 60)


# load data and the tuned params
df_all = pd.read_csv("nhanes_imputed_stacked.csv")
df = df_all[df_all['.imp'] == 1].copy().reset_index(drop=True)

if 'followup_months...6' in df.columns:
    df['followup_months'] = df['followup_months...6']
elif 'followup_months' not in df.columns or df['followup_months'].isna().all():
    df['followup_months'] = df['followup_years'] * 12

folds = pd.read_csv("cv_fold_indices.csv")
df = df.merge(folds, on='SEQN', how='left')

try:
    with open('deephit_best_params.json', 'r') as f:
        bp = json.load(f)
    print("Loaded params from deephit_best_params.json")
except:
    bp = FALLBACK_PARAMS
    print("Using fallback params")


# features, same as part 5
for col in ['race_ethnicity', 'education', 'smoking_status']:
    df[col + '_enc'] = LabelEncoder().fit_transform(df[col].fillna('Unknown'))

feature_cols = [
    'age', 'sex', 'poverty_ratio', 'bmi', 'systolic_bp', 'diastolic_bp',
    'hypertension', 'diabetes', 'chd', 'stroke', 'alcohol_past_year',
    'hba1c', 'total_cholesterol', 'hdl_cholesterol', 'serum_creatinine',
    'fasting_glucose', 'race_ethnicity_enc', 'education_enc', 'smoking_status_enc'
]

# nicer names for the plot labels
feature_labels = [
    'Age', 'Sex (Male)', 'Poverty-Income Ratio', 'BMI',
    'Systolic BP', 'Diastolic BP', 'Hypertension', 'Diabetes',
    'CHD', 'Stroke', 'Alcohol (past yr)', 'HbA1c',
    'Total Cholesterol', 'HDL Cholesterol', 'Serum Creatinine',
    'Fasting Glucose', 'Race/Ethnicity', 'Education', 'Smoking Status'
]

in_features = len(feature_cols)


# discrete time grid + network, copied from part 5
num_risks = 3
labtrans = LabTransDiscreteTime(NUM_DURATIONS)
labtrans.fit(df['followup_months'].values.astype('float32'),
             df['event'].values.astype('int64'))
cuts = labtrans.cuts
n_dur = len(cuts)

def transform_labels(durations, events, cuts):
    idx_dur = np.searchsorted(cuts, durations, side='right') - 1
    return np.clip(idx_dur, 0, len(cuts)-1).astype('int64'), events.astype('int64')

class Reshape3D(torch.nn.Module):
    def __init__(self, nr, nd):
        super().__init__(); self.nr, self.nd = nr, nd
    def forward(self, x):
        return x.view(-1, self.nr, self.nd)

def build_network(in_f, nn, nl, do, nr, nd):
    layers = []
    prev = in_f
    for _ in range(nl):
        layers.extend([torch.nn.Linear(prev, nn), torch.nn.BatchNorm1d(nn),
                        torch.nn.ReLU(), torch.nn.Dropout(do)])
        prev = nn
    layers.append(torch.nn.Linear(prev, nr * nd))
    layers.append(Reshape3D(nr, nd))
    return torch.nn.Sequential(*layers)


# train on folds 2-5, keep fold 1 aside to explain (so the shap values are on
# data the model didn't see)
print("\nTraining model...")
tr = df['fold'] != 1; te = df['fold'] == 1
sc = StandardScaler()
X_train = sc.fit_transform(df.loc[tr, feature_cols].values.astype('float32')).astype('float32')
X_test  = sc.transform(df.loc[te, feature_cols].values.astype('float32')).astype('float32')
y_tr = transform_labels(df.loc[tr, 'followup_months'].values.astype('float32'),
                          df.loc[tr, 'event'].values.astype('int64'), cuts)

n = len(X_train); nv = int(0.15 * n)
pm = np.random.RandomState(SEED + 100).permutation(n)

net = build_network(in_features, bp['num_nodes'], bp['num_layers'],
                    bp['dropout'], num_risks, n_dur)
model = DeepHit(net, tt.optim.Adam, alpha=bp['alpha'], sigma=bp['sigma'],
                duration_index=cuts)
model.optimizer.set_lr(bp['lr'])
model.fit(X_train[pm[nv:]], (y_tr[0][pm[nv:]], y_tr[1][pm[nv:]]),
          batch_size=bp['batch_size'], epochs=300,
          callbacks=[tt.callbacks.EarlyStopping(patience=20)],
          val_data=(X_train[pm[:nv]], (y_tr[0][pm[:nv]], y_tr[1][pm[:nv]])),
          verbose=True)
print("Done.")


# shap needs a scalar output per person, so these wrappers pull out the 10-year
# cif for one cause at a time
t10 = min(np.searchsorted(cuts, 120, side='right') - 1, n_dur - 1)

def pred_cv(X):
    return model.predict_cif(X.astype('float32'))[0][t10, :]

def pred_cancer(X):
    return model.predict_cif(X.astype('float32'))[1][t10, :]


# run kernel shap for each cause. background is a random sample used as the
# reference, ex is the set of people we actually explain.
np.random.seed(SEED)
bg = X_train[np.random.choice(len(X_train), N_BACKGROUND, replace=False)]
ex = X_test[np.random.choice(len(X_test), min(N_EXPLAIN, len(X_test)), replace=False)]

print("\nSHAP for CV death...")
t0 = time.time()
exp_cv = shap.KernelExplainer(pred_cv, bg)
sv_cv = exp_cv.shap_values(ex, nsamples=N_SHAP_SAMPLES)
print("  Done in {:.1f} min".format((time.time() - t0) / 60))

print("SHAP for cancer death...")
t0 = time.time()
exp_ca = shap.KernelExplainer(pred_cancer, bg)
sv_ca = exp_ca.shap_values(ex, nsamples=N_SHAP_SAMPLES)
print("  Done in {:.1f} min".format((time.time() - t0) / 60))


# global importance = mean absolute shap value per feature. then i pull the
# fine-gray coefficients to compare the two rankings.
imp_cv = np.abs(sv_cv).mean(axis=0)
imp_ca = np.abs(sv_ca).mean(axis=0)

fg = pd.read_csv("finegray_coefficients.csv")
# the fine-gray model has separate dummy columns for race/education/smoking,
# whereas deephit has one encoded column each, so this maps the dummies back
# onto the single feature
fg_map = {
    'age':'age', 'sex':'sex', 'poverty_ratio':'poverty_ratio', 'bmi':'bmi',
    'systolic_bp':'systolic_bp', 'diastolic_bp':'diastolic_bp',
    'hypertension':'hypertension', 'diabetes':'diabetes', 'chd':'chd', 'stroke':'stroke',
    'alcohol_past_yr':'alcohol_past_year', 'hba1c':'hba1c',
    'total_chol':'total_cholesterol', 'hdl_chol':'hdl_cholesterol',
    'creatinine':'serum_creatinine', 'fasting_glucose':'fasting_glucose',
    'race_mexican':'race_ethnicity_enc', 'race_nh_black':'race_ethnicity_enc',
    'race_oth_hisp':'race_ethnicity_enc', 'race_other':'race_ethnicity_enc',
    'edu_less_hs':'education_enc', 'edu_hs_ged':'education_enc', 'edu_some_col':'education_enc',
    'smoke_former':'smoking_status_enc', 'smoke_current':'smoking_status_enc'
}

# for fine-gray importance i use distance of the hr from 1 (so a protective and
# a harmful effect of the same size count equally), and for the grouped dummies
# i take the strongest one
def fg_importance(fg_df, fcs, nm):
    imp = {}
    for f in fcs:
        fvs = [k for k, v in nm.items() if v == f]
        hrs = [abs(fg_df[fg_df['variable']==fv]['HR'].values[0]-1)
               for fv in fvs if len(fg_df[fg_df['variable']==fv]) > 0]
        imp[f] = max(hrs) if hrs else 0
    return imp

fg_cv_i = fg_importance(fg[fg['cause']=='CV Death'], feature_cols, fg_map)
fg_ca_i = fg_importance(fg[fg['cause']=='Cancer Death'], feature_cols, fg_map)

idf = pd.DataFrame({
    'feature': feature_cols, 'label': feature_labels,
    'shap_cv': imp_cv, 'shap_cancer': imp_ca,
    'fg_cv': [fg_cv_i[f] for f in feature_cols],
    'fg_cancer': [fg_ca_i[f] for f in feature_cols]
})
for col in ['shap_cv','shap_cancer','fg_cv','fg_cancer']:
    idf[col + '_rank'] = idf[col].rank(ascending=False).astype(int)


# do the two models agree on which features matter? spearman on the ranks.
rho_cv, p_cv = spearmanr(idf['shap_cv_rank'], idf['fg_cv_rank'])
rho_ca, p_ca = spearmanr(idf['shap_cancer_rank'], idf['fg_cancer_rank'])
print("\nSpearman (SHAP vs FG):")
print("  CV Death:     rho={:.3f}, p={:.4f}".format(rho_cv, p_cv))
print("  Cancer Death: rho={:.3f}, p={:.4f}".format(rho_ca, p_ca))


# figures.
# first the side-by-side importance bars, shap on the left and fine-gray on the right
for cause, shap_col, fg_col, rho, p, fname in [
    ('CV Death', 'shap_cv', 'fg_cv', rho_cv, p_cv, 'figure5a_importance_cv.png'),
    ('Cancer Death', 'shap_cancer', 'fg_cancer', rho_ca, p_ca, 'figure5b_importance_cancer.png')
]:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Feature Importance: {} (Spearman rho={:.3f})'.format(cause, rho),
                 fontsize=14, fontweight='bold')
    s = idf.sort_values(shap_col, ascending=True)
    a1.barh(range(len(s)), s[shap_col], color='#E74C3C', alpha=0.8)
    a1.set_yticks(range(len(s))); a1.set_yticklabels(s['label'])
    a1.set_xlabel('Mean |SHAP|'); a1.set_title('DeepHit')
    a2.barh(range(len(s)), s[fg_col], color='#3498DB', alpha=0.8)
    a2.set_yticks(range(len(s))); a2.set_yticklabels([])
    a2.set_xlabel('|HR - 1|'); a2.set_title('Fine-Gray')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

# beeswarm: shows the spread and direction of each feature's effect
for sv, labels_str, fname in [
    (sv_cv, 'CV Death', 'figure5c_beeswarm_cv.png'),
    (sv_ca, 'Cancer Death', 'figure5d_beeswarm_cancer.png')
]:
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, ex, feature_names=feature_labels, show=False, max_display=19)
    plt.title('SHAP Beeswarm: DeepHit {} (10yr CIF)'.format(labels_str), fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

# waterfall plots for the two extremes: the highest and lowest predicted cv risk
for idx_func, title, fname in [
    (np.argmax, 'Highest CV Risk', 'figure5e_waterfall_high.png'),
    (np.argmin, 'Lowest CV Risk', 'figure5f_waterfall_low.png')
]:
    subj = idx_func(pred_cv(ex))
    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        shap.Explanation(values=sv_cv[subj], base_values=exp_cv.expected_value,
                         data=ex[subj], feature_names=feature_labels),
        show=False, max_display=12)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


# save the table and the raw shap arrays so i don't have to recompute them
idf.to_csv('shap_feature_importance.csv', index=False)
np.save('shap_values_cv.npy', sv_cv)
np.save('shap_values_cancer.npy', sv_ca)

print("\nExported: shap_feature_importance.csv, shap_values_cv.npy, shap_values_cancer.npy")
print("Figures: figure5a-f saved.")
print("\ndone.")
