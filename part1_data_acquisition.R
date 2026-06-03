# part 1 - getting the data and building the cohort
# pulls down nhanes 2005-2018, joins it to the nchs mortality files, drops
# people who don't meet the criteria, and writes out the cohort i'll actually
# use for everything else.

library(nhanesA)
library(dplyr)
library(tidyr)
library(haven)
library(survey)
library(readr)

# which cycles to use
# sticking to 2005-2018 (so 7 cycles). the older ones name the lab tables
# differently and i'd basically end up imputing every lab value, not worth it.

cycles <- data.frame(
  cycle  = c("2005-2006","2007-2008","2009-2010","2011-2012",
             "2013-2014","2015-2016","2017-2018"),
  suffix = c("_D","_E","_F","_G","_H","_I","_J"),
  stringsAsFactors = FALSE
)

# little helper so i don't repeat the same loop for every table.
# grabs one table over all 7 cycles and stacks them. the tryCatch is there
# because a couple of tables just don't exist in some cycles and i don't want
# the whole thing to crash when that happens.
download_nhanes_table <- function(base_name, suffixes, cycle_labels, vars) {
  result_list <- list()
  for (i in seq_along(suffixes)) {
    tbl_name <- paste0(base_name, suffixes[i])
    cat("  Downloading:", tbl_name, "\n")
    tryCatch({
      df <- nhanes(tbl_name)
      df <- df %>% select(any_of(c("SEQN", vars)))
      df$cycle <- cycle_labels[i]
      result_list[[i]] <- df
    }, error = function(e) {
      cat("    -> Not available:", conditionMessage(e), "\n")
    })
  }
  bind_rows(result_list)
}


# demographics first (age, sex, race, education, income, survey weights)
cat("\n--- Demographics ---\n")
demo_all <- download_nhanes_table(
  "DEMO", cycles$suffix, cycles$cycle,
  c("RIDAGEYR","RIAGENDR","RIDRETH1","RIDRETH3","DMDEDUC2",
    "INDFMPIR","DMDMARTL","WTMEC2YR","SDMVPSU","SDMVSTRA")
)

# questionnaire stuff: smoking, alcohol, and the self-reported conditions
cat("\n--- Questionnaires ---\n")
smq_all  <- download_nhanes_table("SMQ", cycles$suffix, cycles$cycle,
                                   c("SMQ020","SMQ040"))
alq_all  <- download_nhanes_table("ALQ", cycles$suffix, cycles$cycle,
                                   c("ALQ101","ALQ111","ALQ130"))
bpq_all  <- download_nhanes_table("BPQ", cycles$suffix, cycles$cycle,
                                   c("BPQ020"))
diq_all  <- download_nhanes_table("DIQ", cycles$suffix, cycles$cycle,
                                   c("DIQ010"))
mcq_all  <- download_nhanes_table("MCQ", cycles$suffix, cycles$cycle,
                                   c("MCQ160C","MCQ160F","MCQ220"))

# physical exam: bmi and blood pressure
cat("\n--- Examination ---\n")
bmx_all <- download_nhanes_table("BMX", cycles$suffix, cycles$cycle,
                                  c("BMXBMI","BMXHT","BMXWT"))
bpx_all <- download_nhanes_table("BPX", cycles$suffix, cycles$cycle,
                                  c("BPXSY1","BPXDI1","BPXOSY1","BPXODI1"))

# bloodwork
cat("\n--- Laboratory ---\n")
ghb_all   <- download_nhanes_table("GHB", cycles$suffix, cycles$cycle, c("LBXGH"))
tchol_all <- download_nhanes_table("TCHOL", cycles$suffix, cycles$cycle, c("LBXTC"))
hdl_all   <- download_nhanes_table("HDL", cycles$suffix, cycles$cycle,
                                    c("LBDHDD","LBDHDL","LBDHDDSI"))
biopro_all <- download_nhanes_table("BIOPRO", cycles$suffix, cycles$cycle,
                                     c("LBXSCR","LBXSGL"))
glu_all   <- download_nhanes_table("GLU", cycles$suffix, cycles$cycle,
                                    c("LBXGLU","LBDGLUSI"))


# mortality files from nchs. these are fixed-width .dat files, not the usual
# nhanes tables, so they get downloaded and parsed by hand.
cat("\n--- Mortality files ---\n")

mort_base_url <- "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
mort_files <- data.frame(
  cycle = cycles$cycle,
  filename = c(
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat"
  ),
  stringsAsFactors = FALSE
)

# the column widths and names come straight from the 2019 nchs codebook
mort_widths   <- c(14, 1, 1, 3, 3, 3, 3, 3, 3, 3, 8, 8)
mort_colnames <- c("SEQN","ELIGSTAT","MORTSTAT","UCOD_LEADING",
                   "DIABETES","HYPERTEN","DODQTR","DODYEAR",
                   "WGT_NEW","SA_WGT_NEW","PERMTH_INT","PERMTH_EXM")

mort_list <- list()
for (i in 1:nrow(mort_files)) {
  url <- paste0(mort_base_url, mort_files$filename[i])
  cat("  Downloading:", mort_files$filename[i], "\n")
  tryCatch({
    tmp <- tempfile(fileext = ".dat")
    download.file(url, tmp, mode = "wb", quiet = TRUE)
    df <- read.fwf(tmp, widths = mort_widths, col.names = mort_colnames,
                   strip.white = TRUE, na.strings = c("", " ", "."))
    df$cycle <- mort_files$cycle[i]
    mort_list[[i]] <- df
    file.remove(tmp)
  }, error = function(e) {
    cat("    -> ERROR:", conditionMessage(e), "\n")
  })
}
mort_all <- bind_rows(mort_list)


# now join everything onto the demographics by SEQN. dropping the cycle column
# from each piece first so it doesn't get duplicated a dozen times.
cat("\n--- Merging ---\n")
analytic <- demo_all %>%
  left_join(smq_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(alq_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(bpq_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(diq_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(mcq_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(bmx_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(bpx_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(ghb_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(tchol_all %>% select(-cycle), by = "SEQN") %>%
  left_join(hdl_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(biopro_all %>% select(-cycle), by = "SEQN") %>%
  left_join(glu_all   %>% select(-cycle), by = "SEQN") %>%
  left_join(mort_all  %>% select(-cycle), by = "SEQN")


# a few variables got renamed in the 2017-2018 cycle, so i need to glue the
# old and new versions together. coalesce takes the first non-missing one but
# it errors if a column is missing entirely, hence this wrapper that checks
# which columns actually exist before trying.
safe_coalesce <- function(df, ...) {
  cols <- list(...)
  existing <- cols[cols %in% names(df)]
  if (length(existing) == 0) return(rep(NA_real_, nrow(df)))
  if (length(existing) == 1) return(df[[existing[[1]]]])
  result <- df[[existing[[1]]]]
  for (col in existing[-1]) result <- coalesce(result, df[[col]])
  result
}

analytic <- analytic %>%
  mutate(
    BPXSY1   = safe_coalesce(analytic, "BPXSY1", "BPXOSY1"),
    BPXDI1   = safe_coalesce(analytic, "BPXDI1", "BPXODI1"),
    HDL_CHOL = safe_coalesce(analytic, "LBDHDD", "LBDHDL", "LBDHDDSI"),
    GLUCOSE  = safe_coalesce(analytic, "LBXGLU", "LBDGLUSI", "LBXSGL"),
    ALCOHOL_12 = safe_coalesce(analytic, "ALQ101", "ALQ111"),
    RACE_ETH = RIDRETH1
  )


# eligibility filters, applied one at a time so i can print the count after
# each step and see how many people drop out where.
cat("\n--- Applying eligibility criteria ---\n")
n0 <- nrow(analytic)
analytic <- analytic %>% filter(RIDAGEYR >= 18);               cat("  Age >= 18:      ", nrow(analytic), "\n")
analytic <- analytic %>% filter(ELIGSTAT == 1);                 cat("  Eligible:       ", nrow(analytic), "\n")
analytic <- analytic %>% filter(!is.na(MORTSTAT));              cat("  Valid MORTSTAT:  ", nrow(analytic), "\n")
analytic <- analytic %>% filter(!is.na(PERMTH_EXM) & PERMTH_EXM > 0)
                                                                 cat("  Valid follow-up: ", nrow(analytic), "\n")
analytic <- analytic %>% filter(!is.na(WTMEC2YR) & WTMEC2YR > 0)
                                                                 cat("  Valid MEC wt:   ", nrow(analytic), "\n")
# people who already had cancer at baseline get dropped, otherwise the cancer
# death outcome doesn't really make sense for them
analytic <- analytic %>% filter(is.na(MCQ220) | MCQ220 != 1)
                                                                 cat("  No baseline ca: ", nrow(analytic), "\n")
cat("  Final sample:   ", nrow(analytic), "(excluded", n0 - nrow(analytic), ")\n")


# build the competing event variable.
# 0 = still alive / censored, 1 = died of cardiovascular causes,
# 2 = died of cancer, 3 = died of something else
analytic <- analytic %>%
  mutate(
    event = case_when(
      MORTSTAT == 0                        ~ 0L,
      UCOD_LEADING %in% c(1, 5)           ~ 1L,  # heart disease + stroke grouped as cv
      UCOD_LEADING == 2                    ~ 2L,  # malignant tumours
      UCOD_LEADING %in% c(3,4,6,7,8,9,10) ~ 3L,
      TRUE                                ~ NA_integer_
    ),
    followup_months = PERMTH_EXM,
    followup_years  = PERMTH_EXM / 12
  ) %>%
  filter(!is.na(event))

cat("\nEvent distribution:\n")
print(table(analytic$event))


# the mec weights are per-cycle, so when pooling 7 cycles together you divide
# by the number of cycles
analytic <- analytic %>%
  mutate(WTMEC_POOLED = WTMEC2YR / n_distinct(cycle))


# keep only the columns i need and give them readable names
analytic_final <- analytic %>%
  select(
    SEQN, cycle, WTMEC_POOLED, SDMVPSU, SDMVSTRA,
    event, followup_months, followup_years, MORTSTAT, UCOD_LEADING,
    age = RIDAGEYR, sex = RIAGENDR, race_ethnicity = RACE_ETH,
    education = DMDEDUC2, poverty_ratio = INDFMPIR, marital_status = DMDMARTL,
    bmi = BMXBMI, height_cm = BMXHT, weight_kg = BMXWT,
    systolic_bp = BPXSY1, diastolic_bp = BPXDI1,
    ever_smoked = SMQ020, current_smoker = SMQ040,
    alcohol_12mo = ALCOHOL_12, alcohol_drinks_day = ALQ130,
    hypertension_dx = BPQ020, diabetes_dx = DIQ010,
    chd_dx = MCQ160C, stroke_dx = MCQ160F,
    hba1c = LBXGH, total_cholesterol = LBXTC, hdl_cholesterol = HDL_CHOL,
    serum_creatinine = LBXSCR, fasting_glucose = GLUCOSE
  )


# turn the numeric codes into labelled factors. smoking gets derived from the
# two questions: smq020 (ever smoked 100 cigs) and smq040 (smoke now)
analytic_final <- analytic_final %>%
  mutate(
    sex_label = factor(sex, levels = c(1,2), labels = c("Male","Female")),
    race_label = factor(race_ethnicity, levels = 1:5,
                        labels = c("Mexican American","Other Hispanic",
                                   "NH White","NH Black","Other/Multi")),
    smoking_status = case_when(
      ever_smoked == 2                              ~ "Never",
      ever_smoked == 1 & current_smoker == 3        ~ "Former",
      ever_smoked == 1 & current_smoker %in% c(1,2) ~ "Current",
      TRUE ~ NA_character_
    ),
    event_label = factor(event, levels = 0:3,
                         labels = c("Censored","CV Death","Cancer Death","Other Death"))
  )


# save it
write_csv(analytic_final, "nhanes_analytic_cohort.csv")
cat("\nExported: nhanes_analytic_cohort.csv (", nrow(analytic_final), "rows )\n")
