# part 4 - fine-gray subdistribution hazard model
# fits fine-gray with cmprsk::crr, pools the coefficients across the 20
# imputations using rubin's rules, does 5-fold cv, and writes out the metrics.

library(cmprsk)
library(survival)
library(readr)
library(dplyr)

# load. same followup-column patch as the other scripts in case the merge
# renamed it.
df_all <- read_csv("nhanes_imputed_stacked.csv", show_col_types = FALSE)
if ("followup_months...6" %in% names(df_all)) {
  df_all$followup_months <- df_all$followup_months...6
} else if (all(is.na(df_all$followup_months))) {
  df_all$followup_months <- df_all$followup_years * 12
}

df <- df_all %>% filter(.imp == 1)
cat("Imputation 1:", nrow(df), "rows\n")


# crr wants a plain numeric matrix, so this builds it: continuous variables go
# in as-is, and the categoricals get expanded into dummy columns by hand
# (dropping one level each as the reference)
prepare_covariates <- function(data) {
  X <- data.frame(
    age = data$age, sex = data$sex, poverty_ratio = data$poverty_ratio,
    bmi = data$bmi, systolic_bp = data$systolic_bp, diastolic_bp = data$diastolic_bp,
    hypertension = data$hypertension, diabetes = data$diabetes,
    chd = data$chd, stroke = data$stroke,
    alcohol_past_yr = data$alcohol_past_year,
    hba1c = data$hba1c, total_chol = data$total_cholesterol,
    hdl_chol = data$hdl_cholesterol, creatinine = data$serum_creatinine,
    fasting_glucose = data$fasting_glucose,
    race_mexican  = as.numeric(data$race_ethnicity == "Mexican American"),
    race_oth_hisp = as.numeric(data$race_ethnicity == "Other Hispanic"),
    race_nh_black = as.numeric(data$race_ethnicity == "NH Black"),
    race_other    = as.numeric(data$race_ethnicity == "Other/Multi"),
    edu_less_hs   = as.numeric(data$education == "Less than HS"),
    edu_hs_ged    = as.numeric(data$education == "High school/GED"),
    edu_some_col  = as.numeric(data$education == "Some college"),
    smoke_former  = as.numeric(data$smoking_status == "Former"),
    smoke_current = as.numeric(data$smoking_status == "Current")
  )
  as.matrix(X)
}

X      <- prepare_covariates(df)
ftime  <- df$followup_months
fstatus <- df$event
cat("Covariates:", ncol(X), "\n")


# build the cv folds. doing it stratified by event type so each fold has a
# roughly even share of the (fairly rare) cause-specific deaths, otherwise a
# fold could end up with almost no cancer deaths.
set.seed(2024)
fold_ids <- rep(NA, nrow(df))
for (evt in unique(fstatus)) {
  idx <- which(fstatus == evt)
  fold_ids[sample(idx)] <- rep(1:5, length.out = length(idx))
}

# save these so the python scripts use the exact same splits
write_csv(data.frame(SEQN = df$SEQN, fold = fold_ids), "cv_fold_indices.csv")
cat("Exported: cv_fold_indices.csv\n")


# fit the model on each of the 20 imputations and keep the coefficients + ses,
# then combine with rubin's rules
cat("\n--- Pooling coefficients (m=20) ---\n")

all_coefs <- all_ses <- list()
for (imp_num in 1:20) {
  cat("  Fitting imputation", imp_num, "\n")
  df_imp <- df_all %>% filter(.imp == imp_num)
  df_imp$followup_months <- df_imp$followup_years * 12
  X_imp <- prepare_covariates(df_imp)

  # one fit per cause: failcode=1 is cv death, failcode=2 is cancer death
  fit_cv     <- crr(df_imp$followup_months, df_imp$event, X_imp, failcode=1, cencode=0)
  fit_cancer <- crr(df_imp$followup_months, df_imp$event, X_imp, failcode=2, cencode=0)

  all_coefs[[imp_num]] <- list(cv = fit_cv$coef, cancer = fit_cancer$coef)
  all_ses[[imp_num]]   <- list(cv = sqrt(diag(fit_cv$var)), cancer = sqrt(diag(fit_cancer$var)))
}

# rubin's rules: pooled estimate is just the average. the total variance is the
# within-imputation variance plus the between-imputation variance with the
# (1 + 1/m) correction. then build z, p and the hazard ratios with cis from that.
rubins_pool <- function(coefs_list, ses_list, m = 20) {
  coef_mat <- do.call(rbind, coefs_list)
  se_mat   <- do.call(rbind, ses_list)
  qbar     <- colMeans(coef_mat)            # pooled coefficient
  ubar     <- colMeans(se_mat^2)            # within-imputation variance
  b        <- apply(coef_mat, 2, var)       # between-imputation variance
  total_var <- ubar + (1 + 1/m) * b
  pooled_se <- sqrt(total_var)
  z <- qbar / pooled_se
  p <- 2 * pnorm(-abs(z))
  data.frame(
    variable = names(qbar), coef = round(qbar, 4), se = round(pooled_se, 4),
    z = round(z, 3), p_value = signif(p, 3),
    HR = round(exp(qbar), 3),
    HR_lower = round(exp(qbar - 1.96*pooled_se), 3),
    HR_upper = round(exp(qbar + 1.96*pooled_se), 3),
    stringsAsFactors = FALSE
  )
}

cv_pooled     <- rubins_pool(lapply(all_coefs, `[[`, "cv"),     lapply(all_ses, `[[`, "cv"))
cancer_pooled <- rubins_pool(lapply(all_coefs, `[[`, "cancer"), lapply(all_ses, `[[`, "cancer"))
cv_pooled$cause <- "CV Death"; cancer_pooled$cause <- "Cancer Death"
coef_table <- rbind(cv_pooled, cancer_pooled)
write_csv(coef_table, "finegray_coefficients.csv")

cat("\nCV Death HRs:\n"); print(cv_pooled[, c("variable","HR","HR_lower","HR_upper","p_value")])
cat("\nCancer Death HRs:\n"); print(cancer_pooled[, c("variable","HR","HR_lower","HR_upper","p_value")])


# time-dependent concordance. for each subject who had the event of interest by
# the horizon, compare their predicted risk against everyone who survived longer
# than them. higher predicted risk for the one who died = concordant. ties count
# as half.
compute_ctd <- function(pred_risk, ftime, fstatus, cause, horizon) {
  event_idx <- which(fstatus == cause & ftime <= horizon)
  conc <- disc <- tied <- 0
  for (i in event_idx) {
    comp <- which(ftime > ftime[i])
    d <- pred_risk[i] - pred_risk[comp]
    conc <- conc + sum(d > 0); disc <- disc + sum(d < 0); tied <- tied + 0.5*sum(d == 0)
  }
  total <- conc + disc + tied
  if (total == 0) return(NA)
  (conc + tied) / total
}

# brier score: squared difference between predicted cif and what actually
# happened, at the horizon. lower is better.
compute_brier <- function(pred_cif, ftime, fstatus, cause, horizon) {
  bs <- valid <- 0
  for (i in seq_along(ftime)) {
    if (ftime[i] <= horizon & fstatus[i] == cause) {
      bs <- bs + (1 - pred_cif[i])^2; valid <- valid + 1
    } else if (ftime[i] > horizon) {
      bs <- bs + pred_cif[i]^2; valid <- valid + 1
    }
  }
  if (valid == 0) return(NA)
  bs / valid
}


# 5-fold cv, evaluated at 5 and 10 years
cat("\n--- 5-fold CV ---\n")
horizons_months <- c(60, 120)
horizons_labels <- c("5yr", "10yr")
cv_results <- data.frame()
all_test_preds <- data.frame()

for (fold in 1:5) {
  cat(sprintf("\n  Fold %d/5:\n", fold))
  train_idx <- which(fold_ids != fold)
  test_idx  <- which(fold_ids == fold)

  fit_cv     <- crr(ftime[train_idx], fstatus[train_idx], X[train_idx,], failcode=1, cencode=0)
  fit_cancer <- crr(ftime[train_idx], fstatus[train_idx], X[train_idx,], failcode=2, cencode=0)

  pred_cv     <- predict(fit_cv,     cov1 = X[test_idx,])
  pred_cancer <- predict(fit_cancer, cov1 = X[test_idx,])

  # predict() returns a matrix with the time grid in column 1 and the cifs in
  # the rest, so split those apart
  pred_cv_times  <- pred_cv[,1];  pred_cv_cif  <- pred_cv[,-1, drop=FALSE]
  pred_can_times <- pred_cancer[,1]; pred_can_cif <- pred_cancer[,-1, drop=FALSE]

  for (h in seq_along(horizons_months)) {
    t_h <- horizons_months[h]; t_l <- horizons_labels[h]

    # find the last time point at or before the horizon and read the cif there
    cv_ti  <- max(which(pred_cv_times  <= t_h), na.rm=TRUE)
    can_ti <- max(which(pred_can_times <= t_h), na.rm=TRUE)

    cif_cv_h  <- if(is.finite(cv_ti))  as.numeric(pred_cv_cif[cv_ti,])   else rep(0, length(test_idx))
    cif_can_h <- if(is.finite(can_ti)) as.numeric(pred_can_cif[can_ti,]) else rep(0, length(test_idx))

    ctd_cv  <- compute_ctd(cif_cv_h,  ftime[test_idx], fstatus[test_idx], 1, t_h)
    ctd_can <- compute_ctd(cif_can_h, ftime[test_idx], fstatus[test_idx], 2, t_h)
    bs_cv   <- compute_brier(cif_cv_h,  ftime[test_idx], fstatus[test_idx], 1, t_h)
    bs_can  <- compute_brier(cif_can_h, ftime[test_idx], fstatus[test_idx], 2, t_h)

    cat(sprintf("    %s: C-td(CV)=%.3f, C-td(Cancer)=%.3f\n", t_l, ctd_cv, ctd_can))

    cv_results <- rbind(cv_results, data.frame(
      fold=fold, horizon=t_l, horizon_months=t_h,
      ctd_cv=ctd_cv, ctd_cancer=ctd_can, brier_cv=bs_cv, brier_cancer=bs_can
    ))
  }

  # also keep the 10-year predictions per subject. the python scripts (parts 7
  # and 8) need these to compare against deephit, so they get exported below.
  cv10  <- max(which(pred_cv_times  <= 120), na.rm=TRUE)
  can10 <- max(which(pred_can_times <= 120), na.rm=TRUE)
  all_test_preds <- rbind(all_test_preds, data.frame(
    SEQN = df$SEQN[test_idx], fold = fold,
    followup_months = ftime[test_idx], event = fstatus[test_idx],
    cif_cv_10yr     = if(is.finite(cv10))  as.numeric(pred_cv_cif[cv10,])   else 0,
    cif_cancer_10yr = if(is.finite(can10)) as.numeric(pred_can_cif[can10,]) else 0
  ))
}


# average the folds and save
cat("\n\nFine-Gray CV Results\n")
for (h in horizons_labels) {
  s <- cv_results[cv_results$horizon == h,]
  cat(sprintf("%s: C-td(CV)=%.3f +/- %.3f, C-td(Cancer)=%.3f +/- %.3f\n",
              h, mean(s$ctd_cv), sd(s$ctd_cv), mean(s$ctd_cancer), sd(s$ctd_cancer)))
}

write_csv(cv_results, "finegray_cv_performance.csv")
write_csv(all_test_preds, "finegray_test_predictions.csv")
cat("\nExported: finegray_cv_performance.csv, finegray_test_predictions.csv\n")
