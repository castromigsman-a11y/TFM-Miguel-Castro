# part 2.5 - mice imputation
# reads the clean csv from part 2, runs mice with 20 imputed datasets, has a
# quick look at convergence, and writes out the stacked result.

library(mice)
library(dplyr)
library(readr)
library(ggplot2)
library(lattice)

# load
df <- read_csv("nhanes_clean_for_imputation.csv", show_col_types = FALSE)
cat("Loaded:", nrow(df), "rows x", ncol(df), "columns\n")

# mice needs the categorical variables to actually be factors, otherwise it
# treats them as numbers
df$race_ethnicity <- as.factor(df$race_ethnicity)
df$education      <- as.factor(df$education)
df$smoking_status <- as.factor(df$smoking_status)
df$cycle          <- as.factor(df$cycle)
df$event_factor   <- as.factor(df$event)

# variables that go into the imputation model. i throw in the outcome and the
# cycle as well, not because they need imputing, but because they help predict
# the variables that do (this is the recommended thing to do, otherwise the
# imputations can bias the later survival models)
imp_vars <- c(
  "age","sex","race_ethnicity","education","poverty_ratio",
  "smoking_status","alcohol_past_year","bmi",
  "systolic_bp","diastolic_bp",
  "hypertension","diabetes","chd","stroke",
  "hba1c","total_cholesterol","hdl_cholesterol",
  "serum_creatinine","fasting_glucose",
  "event_factor","followup_months","cycle"
)
imp_df <- df[, imp_vars]

# run mice once with maxit=0 just to grab the default method and predictor
# matrix, then tweak them
ini  <- mice(imp_df, maxit = 0, print = FALSE)
meth <- ini$method
pred <- ini$predictorMatrix

# these ones shouldn't be imputed: the outcome, the time, the cycle, and the
# variables that were already complete (age/sex/race). blank method = leave alone
for (v in c("event_factor","followup_months","cycle","age","sex","race_ethnicity")) {
  meth[v] <- ""
}
diag(pred) <- 0   # a variable shouldn't predict itself

cat("\nMethods:\n")
for (v in names(meth)) {
  if (meth[v] != "") cat("  ", v, "->", meth[v], "\n")
}

# the actual imputation. 20 datasets, 10 iterations each, fixed seed so it
# reproduces. this takes a while.
cat("\nRunning MICE (m=20, maxit=10, seed=2024)...\n")
start_time <- Sys.time()

imp <- mice(imp_df, m = 20, maxit = 10, method = meth,
            predictorMatrix = pred, seed = 2024, printFlag = TRUE)

elapsed <- difftime(Sys.time(), start_time, units = "mins")
cat(sprintf("\nDone in %.1f minutes.\n", as.numeric(elapsed)))

# convergence checks dumped to a pdf. the trace plots should look like random
# noise with no trend, and the density plots compare imputed vs observed values
pdf("mice_diagnostics.pdf", width = 12, height = 8)

cont_imputed <- c("poverty_ratio","bmi","systolic_bp","diastolic_bp",
                  "hba1c","total_cholesterol","hdl_cholesterol",
                  "serum_creatinine","fasting_glucose")
cont_imputed <- cont_imputed[cont_imputed %in% names(meth[meth != ""])]

if (length(cont_imputed) > 0) plot(imp, cont_imputed)
for (v in cont_imputed) {
  tryCatch(densityplot(imp, as.formula(paste("~", v))),
           error = function(e) NULL)
}
dev.off()
cat("Saved: mice_diagnostics.pdf\n")

# mice logs any problems (collinearity etc) here, so print them if there are any
logged <- imp$loggedEvents
if (is.null(logged) || nrow(logged) == 0) {
  cat("No logged events.\n")
} else {
  print(logged)
}

# stack the 20 completed datasets into one long file, with an .imp column
# saying which imputation each row belongs to. also glue back the id/weight/
# outcome columns that weren't part of the imputation.
reattach_vars <- c("SEQN","WTMEC_POOLED","SDMVPSU","SDMVSTRA",
                   "event","followup_months","followup_years",
                   "sex_clean","event_label")

stacked <- data.frame()
for (i in 1:20) {
  completed_i <- complete(imp, action = i)
  completed_i <- cbind(df[, reattach_vars], completed_i)
  completed_i$.imp <- i
  stacked <- rbind(stacked, completed_i)
  if (i %% 5 == 0) cat("  Extracted imputation", i, "of 20\n")
}

# drop the helper columns now that imputation is done
stacked$event_factor <- NULL
stacked$cycle <- NULL

cat("Stacked:", nrow(stacked), "rows\n")

# save the full stacked file plus a single-imputation version for quick checks
write_csv(stacked, "nhanes_imputed_stacked.csv")
cat("Exported: nhanes_imputed_stacked.csv\n")

imp1 <- stacked[stacked$.imp == 1, ]
write_csv(imp1, "nhanes_imputed_single.csv")
cat("Exported: nhanes_imputed_single.csv\n")
