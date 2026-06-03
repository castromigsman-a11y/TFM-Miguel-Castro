# part 2 - cleaning and looking at the missingness
# takes the raw cohort from part 1, sorts out the label encoding mess, looks
# at how much is missing per variable per cycle, throws out anything missing
# more than 40% in a cycle, and saves a clean file ready for mice.

import pandas as pd
import numpy as np

df = pd.read_csv("nhanes_analytic_cohort.csv")
print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")

# nhanesA hands back the text labels rather than the numeric codes, so the
# factor() conversion back in the R script ended up giving me empty columns.
# easiest fix is just to remap the text here.

df['sex_clean'] = df['sex'].str.strip()

race_map = {
    'Mexican American': 'Mexican American',
    'Other Hispanic': 'Other Hispanic',
    'Non-Hispanic White': 'NH White',
    'Non-Hispanic Black': 'NH Black',
    'Other Race - Including Multi-Racial': 'Other/Multi'
}
df['race_clean'] = df['race_ethnicity'].map(race_map)

# education is annoying because the capitalisation isn't consistent between
# cycles, so i map every spelling i found to the same four buckets
edu_map = {
    'Less Than 9th Grade': 'Less than HS',
    'Less than 9th grade': 'Less than HS',
    '9-11th Grade (Includes 12th grade with no diploma)': 'Less than HS',
    '9-11th grade (Includes 12th grade with no diploma)': 'Less than HS',
    'High School Grad/GED or Equivalent': 'High school/GED',
    'High school graduate/GED or equivalent': 'High school/GED',
    'Some College or AA degree': 'Some college',
    'Some college or AA degree': 'Some college',
    'College Graduate or above': 'College+',
    'College graduate or above': 'College+',
    'Refused': None, "Don't Know": None, "Don't know": None
}
df['education_clean'] = df['education'].map(edu_map)

# smoking comes from two questions (smq020 = ever, smq040 = currently)
ever_map = {'Yes': 1, 'No': 0, 'Refused': None, "Don't know": None, "Don't Know": None}
df['ever_smoked_num'] = df['ever_smoked'].map(ever_map).astype('Float64')

current_map = {
    'Every day': 'Current', 'Every day,': 'Current',
    'Some days': 'Current', 'Some days, or': 'Current',
    'Not at all': 'Former', 'Not at all?': 'Former',
    'Refused': None
}
df['current_smoker_clean'] = df['current_smoker'].map(current_map)

# never smoked -> Never. ever smoked + still smokes -> Current.
# ever smoked + stopped -> Former.
df['smoking_clean'] = pd.NA
df.loc[df['ever_smoked_num'] == 0, 'smoking_clean'] = 'Never'
df.loc[(df['ever_smoked_num'] == 1) & (df['current_smoker_clean'] == 'Current'), 'smoking_clean'] = 'Current'
df.loc[(df['ever_smoked_num'] == 1) & (df['current_smoker_clean'] == 'Former'), 'smoking_clean'] = 'Former'

# the yes/no medical conditions. borderline gets 0.5, refused/dk become missing
for old_name, new_name in {'hypertension_dx':'hypertension', 'diabetes_dx':'diabetes',
                            'chd_dx':'chd', 'stroke_dx':'stroke'}.items():
    df[f'{new_name}_num'] = df[old_name].map(
        {'Yes':1.0, 'No':0.0, 'Borderline':0.5, 'Refused':None, "Don't know":None, "Don't Know":None}
    ).astype('Float64')

df['alcohol_12mo_num'] = df['alcohol_12mo'].map(
    {'Yes':1.0, 'No':0.0, 'Refused':None, "Don't know":None, "Don't Know":None}
).astype('Float64')

df['sex_num'] = (df['sex_clean'] == 'Male').astype(int)


# now check the missingness. i look at the worst single cycle for each variable
# rather than the overall rate, because a variable can be fine on average but
# completely absent in one cycle, which would mess up imputation.
analysis_vars = ['age','sex_num','race_clean','education_clean','poverty_ratio',
                 'smoking_clean','alcohol_12mo_num','bmi','systolic_bp','diastolic_bp',
                 'hypertension_num','diabetes_num','chd_num','stroke_num',
                 'hba1c','total_cholesterol','hdl_cholesterol','serum_creatinine',
                 'fasting_glucose']

print("\nMax within-cycle missingness:")
exclude_vars = []
for var in analysis_vars:
    max_miss = 0
    for cycle in sorted(df['cycle'].unique()):
        miss_pct = df[df['cycle'] == cycle][var].isna().mean() * 100
        if miss_pct > max_miss:
            max_miss = miss_pct
    status = "EXCLUDE" if max_miss > 40 else "KEEP"
    if max_miss > 40:
        exclude_vars.append(var)
    print(f"  {var:25s} max={max_miss:5.1f}% -> {status}")

print(f"\nExcluded (>40% in any cycle): {exclude_vars}")


# the pooled weight was originally divided by 10 (assuming 10 cycles), but i'm
# only using 7, so rescale it
df['WTMEC_POOLED'] = df['WTMEC_POOLED'] * 10 / 7


# columns i want to keep in the cleaned file
export_cols = [
    'SEQN','cycle','WTMEC_POOLED','SDMVPSU','SDMVSTRA',
    'event','followup_months','followup_years',
    'age','sex_num','race_clean','education_clean','poverty_ratio',
    'smoking_clean','alcohol_12mo_num','bmi',
    'systolic_bp','diastolic_bp',
    'hypertension_num','diabetes_num','chd_num','stroke_num',
    'hba1c','total_cholesterol','hdl_cholesterol',
    'serum_creatinine','fasting_glucose',
    'sex_clean','event_label'
]
export_cols = [c for c in export_cols if c in df.columns]
df_clean = df[export_cols].copy()

# rename back to the clean names so the rest of the pipeline doesn't have to
# deal with the _num / _clean suffixes
df_clean = df_clean.rename(columns={
    'sex_num':'sex', 'race_clean':'race_ethnicity',
    'education_clean':'education', 'smoking_clean':'smoking_status',
    'alcohol_12mo_num':'alcohol_past_year',
    'hypertension_num':'hypertension', 'diabetes_num':'diabetes',
    'chd_num':'chd', 'stroke_num':'stroke'
})

print(f"\nFinal: {len(df_clean):,} rows x {len(df_clean.columns)} columns")

# quick look at how complete things are before imputing
model_vars = ['age','sex','race_ethnicity','education','poverty_ratio',
              'smoking_status','alcohol_past_year','bmi','systolic_bp','diastolic_bp',
              'hypertension','diabetes','chd','stroke',
              'hba1c','total_cholesterol','hdl_cholesterol','serum_creatinine','fasting_glucose']

complete = df_clean[model_vars].notna().all(axis=1).sum()
print(f"Complete cases: {complete:,} / {len(df_clean):,} ({complete/len(df_clean)*100:.1f}%)")

for var in model_vars:
    pct = df_clean[var].isna().mean() * 100
    print(f"  {var:25s} {pct:5.1f}%")

df_clean.to_csv("nhanes_clean_for_imputation.csv", index=False)
print(f"\nExported: nhanes_clean_for_imputation.csv")
