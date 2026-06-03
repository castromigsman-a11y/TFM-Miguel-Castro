# part 3 - descriptive stats and first look at the data
# makes the baseline table (table 1), the cumulative incidence plots, the
# correlation matrix and a vif check. all on imputation 1.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df_all = pd.read_csv("nhanes_imputed_stacked.csv")
df = df_all[df_all['.imp'] == 1].copy().reset_index(drop=True)

# the followup column sometimes comes back with a weird suffix from the R
# merge, so patch it up if that happened
if 'followup_months...6' in df.columns:
    df['followup_months'] = df['followup_months...6']
elif df['followup_months'].isna().all():
    df['followup_months'] = df['followup_years'] * 12

print(f"Working with imputation 1: {len(df):,} rows")

event_map = {0:'Censored', 1:'CV Death', 2:'Cancer Death', 3:'Other Death'}
df['event_cat'] = df['event'].map(event_map)


# table 1: mean and sd of each continuous variable, split by what people died of
cont_vars = {
    'age':'Age (years)', 'bmi':'BMI (kg/m2)',
    'systolic_bp':'Systolic BP', 'diastolic_bp':'Diastolic BP',
    'poverty_ratio':'PIR', 'hba1c':'HbA1c (%)',
    'total_cholesterol':'Total chol', 'hdl_cholesterol':'HDL chol',
    'serum_creatinine':'Creatinine', 'fasting_glucose':'Fasting glucose'
}

cats = ['Censored','CV Death','Cancer Death','Other Death','Total']
header = f"{'Variable':30s}" + "".join(f" {c:>16s}" for c in cats)
print("\n" + header)
print("-" * 110)

for var, label in cont_vars.items():
    row = f"{label:30s}"
    for cat in cats[:-1]:
        s = df[df['event_cat'] == cat][var]
        row += f" {s.mean():7.1f} +/- {s.std():5.1f}"
    row += f" {df[var].mean():7.1f} +/- {df[var].std():5.1f}"
    print(row)

print(f"\n{'N':30s}" + "".join(f" {(df['event_cat']==c).sum():>16,}" for c in cats[:-1]) + f" {len(df):>16,}")
print(f"Median follow-up: {df['followup_years'].median():.1f} years")


# aalen-johansen estimator for the cumulative incidence.
# can't just use 1 - km here because the competing risks pull people out, so
# this tracks the overall survival and adds up the cause-specific jumps. wrote
# it out by hand to be sure i understood what it was doing.
def aalen_johansen_cif(time, event, cause, weights=None):
    order = np.argsort(time)
    t = np.array(time)[order]
    e = np.array(event)[order]
    w = np.ones(len(t)) if weights is None else np.array(weights)[order]

    unique_times = np.sort(np.unique(t[e > 0]))
    n_risk = np.sum(w)
    cif = np.zeros(len(unique_times))
    surv = 1.0
    cumulative = 0.0

    for i, tj in enumerate(unique_times):
        d_cause = np.sum(w[(t == tj) & (e == cause)])
        d_all   = np.sum(w[(t == tj) & (e > 0)])
        c_j     = np.sum(w[(t == tj) & (e == 0)])
        if n_risk > 0:
            cumulative += surv * (d_cause / n_risk)   # contribution to this cause's cif
            surv *= (1 - d_all / n_risk)              # overall survival drops for any death
        cif[i] = cumulative
        n_risk -= (d_all + c_j)

    return unique_times, cif


# figure 1: the three cifs side by side, with the 5 and 10 year values labelled
cause_labels = {1:'CV Death', 2:'Cancer Death', 3:'Other Death'}
colors = ['#E74C3C','#3498DB','#2ECC71']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Cumulative Incidence Functions by Cause of Death\nNHANES 2005-2018 (n=40,393)",
             fontsize=14, fontweight='bold', y=1.02)

for idx, (cause, label) in enumerate(cause_labels.items()):
    times, cif = aalen_johansen_cif(df['followup_years'].values, df['event'].values, cause)
    axes[idx].plot(times, cif*100, color=colors[idx], linewidth=2)
    axes[idx].set_xlabel('Follow-up (years)')
    axes[idx].set_ylabel('Cumulative Incidence (%)')
    axes[idx].set_title(label, fontweight='bold')
    axes[idx].set_xlim(0, 16)
    axes[idx].grid(True, alpha=0.3)
    for yr in [5, 10]:
        if yr <= times.max():
            ci = cif[np.searchsorted(times, yr, side='right')-1] * 100
            axes[idx].annotate(f'{ci:.1f}%', xy=(yr, ci), fontsize=9, ha='right',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure1_cif_overall.png', dpi=150, bbox_inches='tight')
plt.close()

# same plots again but split by subgroup, to eyeball whether the curves differ.
# first by sex
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("CIF by Sex", fontsize=14, fontweight='bold', y=1.02)
for idx, (cause, label) in enumerate(cause_labels.items()):
    for sex_val in ['Male','Female']:
        mask = df['sex_clean'] == sex_val
        times, cif = aalen_johansen_cif(df.loc[mask,'followup_years'].values,
                                         df.loc[mask,'event'].values, cause)
        axes[idx].plot(times, cif*100, linewidth=2, label=sex_val,
                      linestyle='-' if sex_val=='Male' else '--')
    axes[idx].set_xlabel('Follow-up (years)'); axes[idx].set_ylabel('CIF (%)')
    axes[idx].set_title(label, fontweight='bold'); axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1b_cif_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()

# by age band
df['age_group'] = pd.cut(df['age'], bins=[17,49,64,100], labels=['<50','50-64','65+'])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("CIF by Age Group", fontsize=14, fontweight='bold', y=1.02)
for idx, (cause, label) in enumerate(cause_labels.items()):
    for ag in ['<50','50-64','65+']:
        mask = df['age_group'] == ag
        times, cif = aalen_johansen_cif(df.loc[mask,'followup_years'].values,
                                         df.loc[mask,'event'].values, cause)
        axes[idx].plot(times, cif*100, linewidth=2, label=ag)
    axes[idx].set_xlabel('Follow-up (years)'); axes[idx].set_ylabel('CIF (%)')
    axes[idx].set_title(label, fontweight='bold'); axes[idx].legend(title='Age'); axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1c_cif_by_age.png', dpi=150, bbox_inches='tight')
plt.close()

# by smoking
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("CIF by Smoking Status", fontsize=14, fontweight='bold', y=1.02)
for idx, (cause, label) in enumerate(cause_labels.items()):
    for smoke in ['Never','Former','Current']:
        mask = df['smoking_status'] == smoke
        times, cif = aalen_johansen_cif(df.loc[mask,'followup_years'].values,
                                         df.loc[mask,'event'].values, cause)
        axes[idx].plot(times, cif*100, linewidth=2, label=smoke)
    axes[idx].set_xlabel('Follow-up (years)'); axes[idx].set_ylabel('CIF (%)')
    axes[idx].set_title(label, fontweight='bold'); axes[idx].legend(title='Smoking'); axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1d_cif_by_smoking.png', dpi=150, bbox_inches='tight')
plt.close()


# correlation between the continuous predictors, to spot anything that's going
# to cause collinearity trouble later
cont_model = ['age','bmi','systolic_bp','diastolic_bp','poverty_ratio',
              'hba1c','total_cholesterol','hdl_cholesterol','serum_creatinine','fasting_glucose']
corr = df[cont_model].corr()

print("\nCorrelations |r| > 0.50:")
for i in range(len(cont_model)):
    for j in range(i+1, len(cont_model)):
        r = corr.iloc[i,j]
        if abs(r) > 0.50:
            print(f"  {cont_model[i]} x {cont_model[j]}: r = {r:.3f}")

# heatmap of the same thing
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
short = ['Age','BMI','SBP','DBP','PIR','HbA1c','TC','HDL','SCr','FG']
ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=45, ha='right')
ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
for i in range(len(cont_model)):
    for j in range(len(cont_model)):
        c = 'white' if abs(corr.iloc[i,j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9, color=c)
plt.colorbar(im, label='Pearson r')
plt.title('Correlation Matrix of Continuous Predictors', fontweight='bold')
plt.tight_layout()
plt.savefig('figure2_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# vif as a second collinearity check. standardise the columns and invert the
# correlation matrix, the diagonal gives the variance inflation factors.
# anything over 5 is usually flagged as a problem.
from numpy.linalg import inv
X = df[cont_model].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
XtX_inv = inv((X.T @ X) / len(X))
vifs = np.diag(XtX_inv)
print("\nVIF:")
for v, vif in zip(cont_model, vifs):
    flag = " *** HIGH" if vif > 5 else ""
    print(f"  {v:25s} {vif:.2f}{flag}")

# print the actual cif numbers at 5 and 10 years for the writeup
print("\nCIF at key horizons:")
for cause, label in cause_labels.items():
    times, cif = aalen_johansen_cif(df['followup_years'].values, df['event'].values, cause)
    print(f"  {label}:")
    for yr in [5, 10]:
        if yr <= times.max():
            ci = cif[np.searchsorted(times, yr, side='right')-1] * 100
            print(f"    {yr}-year: {ci:.2f}%")

print("\nDone. Figures saved.")
