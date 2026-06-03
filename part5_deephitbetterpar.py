# part 5 - deephit competing risks model
# deephit from pycox/pytorch, with an optuna hyperparameter search and the
# same 5-fold cv as part 4. i bumped the trial count and epochs up from my
# first run to get steadier results, so this version takes a fair bit longer.
# seeds are fixed so it still reproduces.

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
import warnings, time, os, json

warnings.filterwarnings('ignore')

# settings up top so they're easy to change. the optuna trial count is the
# main thing that drives quality vs runtime; everything else is just giving the
# models a bit more room to converge before early stopping kicks in.
N_OPTUNA_TRIALS   = 300
MAX_EPOCHS_OPTUNA = 150
MAX_EPOCHS_CV     = 400
PATIENCE_OPTUNA   = 15
PATIENCE_CV       = 25
NUM_DURATIONS     = 100
SEED              = 2024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# load + the usual followup patch
df_all = pd.read_csv("nhanes_imputed_stacked.csv")
df = df_all[df_all['.imp'] == 1].copy().reset_index(drop=True)
folds = pd.read_csv("cv_fold_indices.csv")

if 'followup_months...6' in df.columns:
    df['followup_months'] = df['followup_months...6']
elif 'followup_months' not in df.columns or df['followup_months'].isna().all():
    df['followup_months'] = df['followup_years'] * 12

df = df.merge(folds, on='SEQN', how='left')
print("Dataset: {} rows".format(len(df)))


# the neural net needs numbers, so label-encode the three categoricals.
# unlike fine-gray i'm not making dummies here, deephit handles them fine as
# integer codes through the dense layers.
for col in ['race_ethnicity', 'education', 'smoking_status']:
    df[col + '_enc'] = LabelEncoder().fit_transform(df[col].fillna('Unknown'))

feature_cols = [
    'age', 'sex', 'poverty_ratio', 'bmi', 'systolic_bp', 'diastolic_bp',
    'hypertension', 'diabetes', 'chd', 'stroke', 'alcohol_past_year',
    'hba1c', 'total_cholesterol', 'hdl_cholesterol', 'serum_creatinine',
    'fasting_glucose', 'race_ethnicity_enc', 'education_enc', 'smoking_status_enc'
]
in_features = len(feature_cols)


# deephit works on discrete time, so chop the follow-up into 100 intervals
num_risks = 3
labtrans = LabTransDiscreteTime(NUM_DURATIONS)
labtrans.fit(df['followup_months'].values.astype('float32'),
             df['event'].values.astype('int64'))
cuts = labtrans.cuts
n_dur = len(cuts)


# map each follow-up time to the index of the interval it falls in
def transform_labels(durations, events, cuts):
    idx_dur = np.searchsorted(cuts, durations, side='right') - 1
    idx_dur = np.clip(idx_dur, 0, len(cuts) - 1).astype('int64')
    return idx_dur, events.astype('int64')


# the network outputs one value per (risk, time-interval) pair, so the final
# layer is num_risks * num_durations wide and then i reshape it into a 3d block
class Reshape3D(torch.nn.Module):
    def __init__(self, num_risks, num_durations):
        super().__init__()
        self.num_risks = num_risks
        self.num_durations = num_durations

    def forward(self, x):
        return x.view(-1, self.num_risks, self.num_durations)


def build_network(in_features, num_nodes, num_layers, dropout,
                  num_risks, num_durations):
    layers = []
    prev_size = in_features
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(prev_size, num_nodes))
        layers.append(torch.nn.BatchNorm1d(num_nodes))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        prev_size = num_nodes
    layers.append(torch.nn.Linear(prev_size, num_risks * num_durations))
    layers.append(Reshape3D(num_risks, num_durations))
    return torch.nn.Sequential(*layers)


# same concordance and brier definitions as the fine-gray script, just in numpy
def compute_ctd(cif_cause, times_grid, durations, events, cause, horizon):
    t_idx = min(np.searchsorted(times_grid, horizon, side='right') - 1,
                len(times_grid) - 1)
    pred = cif_cause[t_idx, :]
    ev_idx = np.where((events == cause) & (durations <= horizon))[0]
    if len(ev_idx) == 0:
        return np.nan
    conc = disc = tied = 0
    for i in ev_idx:
        d = pred[i] - pred[durations > durations[i]]
        conc += np.sum(d > 0); disc += np.sum(d < 0); tied += 0.5 * np.sum(d == 0)
    total = conc + disc + tied
    return (conc + tied) / total if total > 0 else np.nan


def compute_brier(cif_cause, times_grid, durations, events, cause, horizon):
    t_idx = min(np.searchsorted(times_grid, horizon, side='right') - 1,
                len(times_grid) - 1)
    p = cif_cause[t_idx, :]
    m_e = (durations <= horizon) & (events == cause)
    m_s = durations > horizon
    bs = np.sum((1 - p[m_e])**2) + np.sum(p[m_s]**2)
    v = m_e.sum() + m_s.sum()
    return bs / v if v > 0 else np.nan


# hyperparameter search. i hold out fold 1 as the validation set for tuning,
# train on the rest, and let optuna maximise 10-year cv-death concordance. it's
# set up as a minimisation of -ctd because that's what the sampler expects.
print("\nOptuna search ({} trials)...".format(N_OPTUNA_TRIALS))

hp_tr = ~df['fold'].isin([1]); hp_te = df['fold'] == 1
sc_hp = StandardScaler()
X_hp_tr = sc_hp.fit_transform(df.loc[hp_tr, feature_cols].values.astype('float32')).astype('float32')
X_hp_te = sc_hp.transform(df.loc[hp_te, feature_cols].values.astype('float32')).astype('float32')
dur_hp_te = df.loc[hp_te, 'followup_months'].values.astype('float32')
evt_hp_te = df.loc[hp_te, 'event'].values.astype('int64')
y_hp_tr = transform_labels(df.loc[hp_tr, 'followup_months'].values.astype('float32'),
                            df.loc[hp_tr, 'event'].values.astype('int64'), cuts)

# carve out 15% of the training data as an internal validation set for early
# stopping during each trial
n_hp = len(X_hp_tr); n_hp_val = int(0.15 * n_hp)
perm_hp = np.random.RandomState(SEED).permutation(n_hp)
Xit, Xiv = X_hp_tr[perm_hp[n_hp_val:]], X_hp_tr[perm_hp[:n_hp_val]]
yit = (y_hp_tr[0][perm_hp[n_hp_val:]], y_hp_tr[1][perm_hp[n_hp_val:]])
yiv = (y_hp_tr[0][perm_hp[:n_hp_val]], y_hp_tr[1][perm_hp[:n_hp_val]])

best_ctd_g = -np.inf
best_params_g = None

def objective(trial):
    global best_ctd_g, best_params_g
    nl = trial.suggest_int('num_layers', 1, 4)
    nn = trial.suggest_categorical('num_nodes', [32, 64, 128, 256])
    do = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical('batch_size', [128, 256, 512])
    al = trial.suggest_float('alpha', 0.1, 1.0, step=0.1)
    si = trial.suggest_float('sigma', 0.1, 1.0, step=0.1)

    net = build_network(in_features, nn, nl, do, num_risks, n_dur)
    model = DeepHit(net, tt.optim.Adam, alpha=al, sigma=si, duration_index=cuts)
    model.optimizer.set_lr(lr)
    try:
        model.fit(Xit, yit, batch_size=bs, epochs=MAX_EPOCHS_OPTUNA,
                  callbacks=[tt.callbacks.EarlyStopping(patience=PATIENCE_OPTUNA)],
                  val_data=(Xiv, yiv), verbose=False)
        cif = model.predict_cif(X_hp_te)
        ctd = compute_ctd(cif[0], cuts, dur_hp_te, evt_hp_te, 1, 120)
        if not np.isnan(ctd):
            if ctd > best_ctd_g:
                best_ctd_g = ctd
                best_params_g = trial.params.copy()
                print("  Trial {:3d}: C-td={:.4f} *".format(trial.number, ctd))
            return -ctd
    except:
        # some configs blow up (nan loss etc), just skip them
        pass
    return float('inf')

t0 = time.time()
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

# if every trial somehow failed, fall back to something sensible so the rest of
# the script still runs
if best_params_g is None:
    bp = {'num_layers':2, 'num_nodes':128, 'dropout':0.2,
          'lr':0.001, 'batch_size':256, 'alpha':0.5, 'sigma':0.5}
else:
    bp = best_params_g

print("\nOptuna done in {:.1f} min".format((time.time() - t0) / 60))
for k, v in bp.items():
    print("  {:15s} = {}".format(k, v))


# now refit with the best params across all 5 folds for the real evaluation
print("\n5-fold CV...")
horizons = {'5yr': 60, '10yr': 120}
cv_results, all_preds = [], []

for fold in range(1, 6):
    print("\n  Fold {}/5:".format(fold))
    tr = df['fold'] != fold; te = df['fold'] == fold
    sc = StandardScaler()
    Xtr = sc.fit_transform(df.loc[tr, feature_cols].values.astype('float32')).astype('float32')
    Xte = sc.transform(df.loc[te, feature_cols].values.astype('float32')).astype('float32')
    d_tr = df.loc[tr, 'followup_months'].values.astype('float32')
    e_tr = df.loc[tr, 'event'].values.astype('int64')
    d_te = df.loc[te, 'followup_months'].values.astype('float32')
    e_te = df.loc[te, 'event'].values.astype('int64')
    y_tr = transform_labels(d_tr, e_tr, cuts)

    # the +fold in the seed means each fold gets a different but reproducible
    # validation split
    n = len(Xtr); nv = int(0.15 * n)
    pm = np.random.RandomState(SEED + fold).permutation(n)
    Xt, Xv = Xtr[pm[nv:]], Xtr[pm[:nv]]
    yt = (y_tr[0][pm[nv:]], y_tr[1][pm[nv:]])
    yv = (y_tr[0][pm[:nv]], y_tr[1][pm[:nv]])

    net = build_network(in_features, bp['num_nodes'], bp['num_layers'],
                        bp['dropout'], num_risks, n_dur)
    model = DeepHit(net, tt.optim.Adam, alpha=bp['alpha'], sigma=bp['sigma'],
                    duration_index=cuts)
    model.optimizer.set_lr(bp['lr'])
    log = model.fit(Xt, yt, batch_size=bp['batch_size'], epochs=MAX_EPOCHS_CV,
                    callbacks=[tt.callbacks.EarlyStopping(patience=PATIENCE_CV)],
                    val_data=(Xv, yv), verbose=False)

    # just for the log, how many epochs it actually trained before stopping
    try:
        n_ep = len(list(log.monitors.values())[0].scores)
    except:
        n_ep = "?"
    print("    Epochs: {}".format(n_ep))

    cif = model.predict_cif(Xte)

    for hl, hm in horizons.items():
        for c, cl in [(1, 'CV Death'), (2, 'Cancer Death')]:
            ctd = compute_ctd(cif[c-1], cuts, d_te, e_te, c, hm)
            bs = compute_brier(cif[c-1], cuts, d_te, e_te, c, hm)
            cv_results.append({'fold':fold, 'horizon':hl, 'cause':cl,
                               'ctd':ctd, 'brier':bs})
            print("    {} {}: C-td={:.3f}, Brier={:.4f}".format(hl, cl, ctd, bs))

    # store the 10-year predictions for the calibration / stage-1 scripts
    ti = min(np.searchsorted(cuts, 120, side='right') - 1, n_dur - 1)
    all_preds.append(pd.DataFrame({
        'SEQN': df.loc[te, 'SEQN'].values, 'fold': fold,
        'followup_months': d_te, 'event': e_te,
        'cif_cv_10yr': cif[0][ti, :], 'cif_cancer_10yr': cif[1][ti, :]
    }))


# summarise and write out
cv_df = pd.DataFrame(cv_results)

print("\n" + "=" * 60)
print("DeepHit Summary:")
for h in ['5yr', '10yr']:
    for cause in ['CV Death', 'Cancer Death']:
        sub = cv_df[(cv_df['horizon'] == h) & (cv_df['cause'] == cause)]
        print("  {} {}: C-td = {:.3f} +/- {:.3f}".format(
            h, cause, sub['ctd'].mean(), sub['ctd'].std()))

# if the fine-gray results are already on disk, print a quick side-by-side
fg_path = 'finegray_cv_performance.csv'
if os.path.exists(fg_path):
    fg = pd.read_csv(fg_path)
    print("\nHead-to-head:")
    for h in ['5yr', '10yr']:
        fh = fg[fg['horizon'] == h]; dh = cv_df[cv_df['horizon'] == h]
        for cs, fc in [('CV', 'ctd_cv'), ('Cancer', 'ctd_cancer')]:
            fv = fh[fc].mean()
            dv = dh[dh['cause'] == cs + ' Death']['ctd'].mean()
            print("  C-td {} ({}): FG={:.3f}, DH={:.3f}, diff={:+.3f}".format(
                cs, h, fv, dv, dv - fv))

cv_df.to_csv('deephit_cv_performance.csv', index=False)
pd.concat(all_preds).to_csv('deephit_test_predictions.csv', index=False)
# save the chosen params so parts 6, 7 and 8 can reuse them without re-tuning
with open('deephit_best_params.json', 'w') as f:
    json.dump(bp, f, indent=2)

print("\nExported: deephit_cv_performance.csv, deephit_test_predictions.csv, deephit_best_params.json")
print("Runtime: {:.1f} min".format((time.time() - t0) / 60))
