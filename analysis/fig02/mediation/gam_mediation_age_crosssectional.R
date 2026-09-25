############################################################
# GAM-based mediation analysis for cross-sectional age data
# Exposure: age (continuous, e.g., 6-18 years)
# Mediator: selected continuous variable M
# Outcome: selected continuous variable outcome
# Models:
#   1) No age moderation of M -> outcome: Y ~ s(age) + M + covariates
#   2) Age-moderated M -> outcome: Y ~ s(age) + M + s(age, by = M) + covariates
# Estimation:
#   Counterfactual g-computation with nonparametric bootstrap CIs
############################################################

## =========================
## 0. User settings
## =========================

RUN_SIMULATION <- FALSE

# Used only when RUN_SIMULATION == FALSE
input_csv <- "/public/home/dingrui/fmri_analysis/data/beh/er_profiles_new.csv"

# Column names in your CSV
id_var       <- "sub_id"
age_var      <- "age"
gender_var   <- "sex"
mediator_var <- "reappraisal_success"
outcome_var  <- "ERQ_CR"

# Pre-treatment covariates to adjust for. For your current dataset, gender is the default.
# Do not include age, mediator, outcome, or sub_id here.
covariates <- c("sex")

# Age contrasts. Effects are interpreted as changing age from age0 to age1.
age_contrasts <- data.frame(
  age0 = c(8, 12, 16, 8, 8, 11),
  age1 = c(12, 16, 19, 19, 11, 19)
)

# Spline basis dimension for age. For 6-18 years, 5-7 is usually a reasonable start.
k_age <- 7

# Bootstrap settings. Increase to 1000-5000 for manuscript-level inference.
BOOTSTRAP_B <- if (RUN_SIMULATION) 100 else 1000
seed <- 20260620

# Output folder
outdir <- "/public/home/dingrui/tests/gam_mediation/reappraisal_success"

## =========================
## 1. Package setup
## =========================

required_pkgs <- c("mgcv")
for (p in required_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}
library(mgcv)

if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

## =========================
## 2. Helper functions
## =========================

simulate_mediation_data <- function(n = 700, seed = 1, age_moderated_slope = TRUE) {
  set.seed(seed)
  age <- runif(n, min = 6, max = 18)
  gender <- factor(sample(c("female", "male"), size = n, replace = TRUE))
  g_num <- ifelse(gender == "female", 0.5, -0.5)
  age_sc <- (age - 12) / 6

  # Nonlinear age -> M relation
  f_age_m <- 1.20 * sin(pi * (age - 6) / 12) - 0.80 * age_sc^2 + 0.15 * age_sc
  M <- f_age_m + 0.25 * g_num + rnorm(n, mean = 0, sd = 0.55)

  # Linear M -> outcome relation; optionally the M slope varies smoothly with age.
  slope_M <- if (age_moderated_slope) 0.85 + 0.35 * age_sc else 0.85
  direct_age <- 0.25 * age_sc + 0.25 * sin(1.5 * pi * (age - 6) / 12)
  outcome <- direct_age + slope_M * M + 0.20 * g_num + rnorm(n, mean = 0, sd = 0.65)

  data.frame(
    sub_id = sprintf("sub_%04d", seq_len(n)),
    age = round(age, 4),
    gender = gender,
    M = round(M, 5),
    outcome = round(outcome, 5),
    stringsAsFactors = FALSE
  )
}

mode_value <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

clean_and_validate_data <- function(dat, id_var, age_var, gender_var, mediator_var, outcome_var, covariates) {
  required_cols <- unique(c(id_var, age_var, gender_var, mediator_var, outcome_var, covariates))
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # Standardize internal names for formulas.
  # The script keeps age/gender as canonical internal names.
  if (age_var != "age") names(dat)[names(dat) == age_var] <- "age"
  if (gender_var != "gender") names(dat)[names(dat) == gender_var] <- "gender"
  if (mediator_var != "M") names(dat)[names(dat) == mediator_var] <- "M"
  if (outcome_var != "outcome") names(dat)[names(dat) == outcome_var] <- "outcome"

  covariates_internal <- covariates
  covariates_internal[covariates_internal == age_var] <- "age"
  covariates_internal[covariates_internal == gender_var] <- "gender"
  covariates_internal[covariates_internal == mediator_var] <- "M"
  covariates_internal[covariates_internal == outcome_var] <- "outcome"
  covariates_internal <- setdiff(unique(covariates_internal), c("age", "M", "outcome"))

  dat$age <- as.numeric(dat$age)
  dat$M <- as.numeric(dat$M)
  dat$outcome <- as.numeric(dat$outcome)
  dat$gender <- factor(dat$gender)

  model_cols <- unique(c("age", "gender", "M", "outcome", covariates_internal))
  dat <- dat[complete.cases(dat[, model_cols, drop = FALSE]), , drop = FALSE]

  if (nrow(dat) < 50) warning("After removing missing data, n < 50. GAM mediation estimates may be unstable.")
  if (any(dat$age < 6 | dat$age > 18)) warning("Some ages are outside the expected 6-18 range.")
  if (length(unique(dat$age)) < 8) warning("Age has few unique values. Consider reducing k_age.")

  list(data = dat, covariates = covariates_internal)
}

rhs_with_covariates <- function(main_terms, covariates) {
  terms <- c(main_terms, covariates)
  paste(terms, collapse = " + ")
}

fit_model_set <- function(dat, covariates, k_age = 6) {
  f_m <- as.formula(paste0(
    "M ~ ",
    rhs_with_covariates(c(sprintf("s(age, k = %d, bs = 'cr')", k_age)), covariates)
  ))

  f_y_no_moderation <- as.formula(paste0(
    "outcome ~ ",
    rhs_with_covariates(c(sprintf("s(age, k = %d, bs = 'cr')", k_age), "M"), covariates)
  ))

  # Varying-coefficient GAM: the slope of M is allowed to vary smoothly with age.
  # This keeps the M -> outcome association linear in M at each age.
  f_y_age_moderation <- as.formula(paste0(
    "outcome ~ ",
    rhs_with_covariates(c(sprintf("s(age, k = %d, bs = 'cr')", k_age),
                          "M",
                          sprintf("s(age, by = M, k = %d, bs = 'cr')", k_age)), covariates)
  ))

  m_mod <- mgcv::gam(f_m, data = dat, method = "REML", select = TRUE)
  y_no <- mgcv::gam(f_y_no_moderation, data = dat, method = "REML", select = TRUE)
  y_age_mod <- mgcv::gam(f_y_age_moderation, data = dat, method = "REML", select = TRUE)

  list(
    mediator = m_mod,
    outcome = list(
      no_age_moderation = y_no,
      age_moderation = y_age_mod
    ),
    formulas = list(
      mediator = f_m,
      no_age_moderation = f_y_no_moderation,
      age_moderation = f_y_age_moderation
    )
  )
}

estimate_effects_one_model <- function(m_model, y_model, dat, age0, age1) {
  d_a0 <- dat
  d_a1 <- dat
  d_a0$age <- age0
  d_a1$age <- age1

  M_a0 <- as.numeric(predict(m_model, newdata = d_a0, type = "response"))
  M_a1 <- as.numeric(predict(m_model, newdata = d_a1, type = "response"))

  y_a0_Ma0 <- d_a0
  y_a0_Ma1 <- d_a0
  y_a1_Ma0 <- d_a1
  y_a1_Ma1 <- d_a1

  y_a0_Ma0$M <- M_a0
  y_a0_Ma1$M <- M_a1
  y_a1_Ma0$M <- M_a0
  y_a1_Ma1$M <- M_a1

  Y_a0_Ma0 <- as.numeric(predict(y_model, newdata = y_a0_Ma0, type = "response"))
  Y_a0_Ma1 <- as.numeric(predict(y_model, newdata = y_a0_Ma1, type = "response"))
  Y_a1_Ma0 <- as.numeric(predict(y_model, newdata = y_a1_Ma0, type = "response"))
  Y_a1_Ma1 <- as.numeric(predict(y_model, newdata = y_a1_Ma1, type = "response"))

  # Counterfactual mediation quantities for continuous exposure age.
  ACME_at_age0 <- mean(Y_a0_Ma1 - Y_a0_Ma0)
  ACME_at_age1 <- mean(Y_a1_Ma1 - Y_a1_Ma0)
  ADE_Ma0      <- mean(Y_a1_Ma0 - Y_a0_Ma0)
  ADE_Ma1      <- mean(Y_a1_Ma1 - Y_a0_Ma1)
  total        <- mean(Y_a1_Ma1 - Y_a0_Ma0)

  safe_ratio <- function(num, den) ifelse(abs(den) < 1e-10, NA_real_, num / den)

  c(
    ACME_at_age0 = ACME_at_age0,
    ACME_at_age1 = ACME_at_age1,
    ACME_avg     = mean(c(ACME_at_age0, ACME_at_age1)),
    ADE_Ma0      = ADE_Ma0,
    ADE_Ma1      = ADE_Ma1,
    ADE_avg      = mean(c(ADE_Ma0, ADE_Ma1)),
    Total_effect = total,
    Prop_med_avg = safe_ratio(mean(c(ACME_at_age0, ACME_at_age1)), total)
  )
}

estimate_effect_grid <- function(fits, dat, age_contrasts) {
  rows <- list()
  idx <- 1
  for (model_name in names(fits$outcome)) {
    y_model <- fits$outcome[[model_name]]
    for (i in seq_len(nrow(age_contrasts))) {
      age0 <- age_contrasts$age0[i]
      age1 <- age_contrasts$age1[i]
      est <- estimate_effects_one_model(fits$mediator, y_model, dat, age0, age1)
      rows[[idx]] <- data.frame(
        model = model_name,
        contrast = paste0(age0, "_to_", age1),
        age0 = age0,
        age1 = age1,
        as.data.frame(as.list(est)),
        row.names = NULL,
        check.names = FALSE
      )
      idx <- idx + 1
    }
  }
  do.call(rbind, rows)
}

wide_to_long <- function(df) {
  id_cols <- intersect(c("boot", "model", "contrast", "age0", "age1"), names(df))
  effect_cols <- setdiff(names(df), id_cols)
  out <- vector("list", nrow(df) * length(effect_cols))
  idx <- 1
  for (i in seq_len(nrow(df))) {
    for (effect in effect_cols) {
      row <- df[i, id_cols, drop = FALSE]
      row$effect <- effect
      row$value <- as.numeric(df[i, effect])
      out[[idx]] <- row
      idx <- idx + 1
    }
  }
  do.call(rbind, out)
}

summarize_bootstrap <- function(point_df, boot_df) {
  point_long <- wide_to_long(point_df)
  names(point_long)[names(point_long) == "value"] <- "estimate"

  boot_long <- wide_to_long(boot_df)

  key_cols <- c("model", "contrast", "age0", "age1", "effect")
  summaries <- vector("list", nrow(point_long))

  for (i in seq_len(nrow(point_long))) {
    key <- point_long[i, key_cols, drop = FALSE]
    sel <- rep(TRUE, nrow(boot_long))
    for (k in key_cols) sel <- sel & boot_long[[k]] == key[[k]]
    vals <- boot_long$value[sel]
    vals <- vals[is.finite(vals)]

    if (length(vals) >= 10) {
      ci <- as.numeric(quantile(vals, probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE))
      p_boot <- 2 * min(mean(vals <= 0), mean(vals >= 0))
      p_boot <- min(p_boot, 1)
      boot_se <- sd(vals, na.rm = TRUE)
    } else {
      ci <- c(NA_real_, NA_real_)
      p_boot <- NA_real_
      boot_se <- NA_real_
    }

    summaries[[i]] <- data.frame(
      point_long[i, key_cols, drop = FALSE],
      estimate = point_long$estimate[i],
      boot_se = boot_se,
      ci_low = ci[1],
      ci_high = ci[2],
      p_boot = p_boot,
      n_boot_success = length(vals),
      row.names = NULL
    )
  }

  do.call(rbind, summaries)
}

bootstrap_mediation <- function(dat, covariates, age_contrasts, k_age = 6, B = 200, seed = 1) {
  set.seed(seed)

  message("Fitting point-estimate models...")
  point_fits <- fit_model_set(dat, covariates, k_age)
  point_effects <- estimate_effect_grid(point_fits, dat, age_contrasts)

  boot_rows <- vector("list", B)
  success <- 0

  message("Running nonparametric bootstrap: B = ", B)
  for (b in seq_len(B)) {
    idx <- sample(seq_len(nrow(dat)), size = nrow(dat), replace = TRUE)
    db <- dat[idx, , drop = FALSE]

    # Preserve original factor levels for stable prediction/model matrices.
    if ("gender" %in% names(db)) db$gender <- factor(db$gender, levels = levels(dat$gender))
    for (cv in covariates) {
      if (is.factor(dat[[cv]])) db[[cv]] <- factor(db[[cv]], levels = levels(dat[[cv]]))
    }

    fit_b <- try(fit_model_set(db, covariates, k_age), silent = TRUE)
    if (inherits(fit_b, "try-error")) next

    eff_b <- try(estimate_effect_grid(fit_b, db, age_contrasts), silent = TRUE)
    if (inherits(eff_b, "try-error")) next

    eff_b$boot <- b
    boot_rows[[b]] <- eff_b
    success <- success + 1

    if (b %% 25 == 0) message("  Bootstrap iteration ", b, "/", B, "; successful = ", success)
  }

  boot_df <- do.call(rbind, boot_rows[!vapply(boot_rows, is.null, logical(1))])
  if (is.null(boot_df) || nrow(boot_df) == 0) stop("All bootstrap iterations failed.")

  results <- summarize_bootstrap(point_effects, boot_df)
  list(point_fits = point_fits, point_effects = point_effects, boot_effects = boot_df, results = results)
}

write_model_report <- function(fits, dat, output_file) {
  sink(output_file)
  cat("GAM mediation model report\n")
  cat("==========================\n\n")
  cat("N = ", nrow(dat), "\n", sep = "")
  cat("Age range = ", round(min(dat$age), 3), " to ", round(max(dat$age), 3), "\n\n", sep = "")

  cat("Mediator model formula:\n")
  print(fits$formulas$mediator)
  cat("\n")
  print(summary(fits$mediator))
  cat("\nApproximate k-check / diagnostics:\n")
  print(try(gam.check(fits$mediator, rep = 0), silent = TRUE))

  for (nm in names(fits$outcome)) {
    cat("\n\nOutcome model: ", nm, "\n", sep = "")
    cat("Formula:\n")
    print(fits$formulas[[nm]])
    cat("\n")
    print(summary(fits$outcome[[nm]]))
    cat("\nApproximate k-check / diagnostics:\n")
    print(try(gam.check(fits$outcome[[nm]], rep = 0), silent = TRUE))
  }

  cat("\n\nOutcome model comparison\n")
  cat("------------------------\n")
  print(AIC(fits$outcome$no_age_moderation, fits$outcome$age_moderation))
  cat("\nApproximate nested-model comparison, if valid for fitted models:\n")
  print(try(anova(fits$outcome$no_age_moderation, fits$outcome$age_moderation, test = "F"), silent = TRUE))

  sink()
}

make_plots <- function(dat, fits, results, outdir) {
  # Plot age -> mediator trajectory from the mediator GAM.
  pdf(file.path(outdir, "plot_age_to_M.pdf"), width = 7, height = 5)
  plot(dat$age, dat$M, pch = 16, col = rgb(0, 0, 0, 0.25),
       xlab = "Age", ylab = "Mediator M",
       main = "Nonlinear age -> M trajectory")

  age_seq <- seq(min(dat$age), max(dat$age), length.out = 200)
  nd <- dat[rep(1, length(age_seq)), , drop = FALSE]
  nd$age <- age_seq
  if ("gender" %in% names(nd)) nd$gender <- factor(mode_value(dat$gender), levels = levels(dat$gender))
  pred_M <- predict(fits$mediator, newdata = nd, type = "response")
  lines(age_seq, pred_M, lwd = 3)
  dev.off()

  # Plot ACME_avg across age contrasts.
  pdf(file.path(outdir, "plot_ACME_avg_by_age_contrast.pdf"), width = 9, height = 5)
  acme <- results[results$effect == "ACME_avg", , drop = FALSE]
  acme$x <- seq_len(nrow(acme))
  ylim <- range(c(acme$ci_low, acme$ci_high, acme$estimate), na.rm = TRUE)
  plot(acme$x, acme$estimate, pch = 16, ylim = ylim, xaxt = "n",
       xlab = "Age contrast", ylab = "ACME average",
       main = "Average mediated effect by age contrast")
  axis(1, at = acme$x, labels = paste(acme$model, acme$contrast, sep = "\n"), cex.axis = 0.65)
  arrows(acme$x, acme$ci_low, acme$x, acme$ci_high, angle = 90, code = 3, length = 0.04)
  abline(h = 0, lty = 2)
  dev.off()
}

## =========================
## 3. Load or simulate data
## =========================

if (RUN_SIMULATION) {
  dat_raw <- simulate_mediation_data(n = 700, seed = seed, age_moderated_slope = TRUE)
  simulated_csv <- file.path(outdir, "simulated_age_mediation_data.csv")
  write.csv(dat_raw, simulated_csv, row.names = FALSE)
  message("Simulated data written to: ", simulated_csv)
} else {
  dat_raw <- read.csv(input_csv, stringsAsFactors = FALSE)
}

cleaned <- clean_and_validate_data(
  dat = dat_raw,
  id_var = id_var,
  age_var = age_var,
  gender_var = gender_var,
  mediator_var = mediator_var,
  outcome_var = outcome_var,
  covariates = covariates
)

dat <- cleaned$data
covariates <- cleaned$covariates

## =========================
## 4. Run analysis
## =========================

analysis <- bootstrap_mediation(
  dat = dat,
  covariates = covariates,
  age_contrasts = age_contrasts,
  k_age = k_age,
  B = BOOTSTRAP_B,
  seed = seed
)

## =========================
## 5. Save outputs
## =========================

write.csv(analysis$results,
          file.path(outdir, "mediation_results_bootstrap.csv"),
          row.names = FALSE)
write.csv(analysis$point_effects,
          file.path(outdir, "mediation_point_estimates_wide.csv"),
          row.names = FALSE)
write.csv(analysis$boot_effects,
          file.path(outdir, "mediation_bootstrap_draws_wide.csv"),
          row.names = FALSE)

model_compare <- data.frame(
  model = c("no_age_moderation", "age_moderation"),
  AIC = c(AIC(analysis$point_fits$outcome$no_age_moderation),
          AIC(analysis$point_fits$outcome$age_moderation)),
  dev_explained = c(summary(analysis$point_fits$outcome$no_age_moderation)$dev.expl,
                    summary(analysis$point_fits$outcome$age_moderation)$dev.expl),
  adj_r_squared = c(summary(analysis$point_fits$outcome$no_age_moderation)$r.sq,
                    summary(analysis$point_fits$outcome$age_moderation)$r.sq)
)
model_compare$delta_AIC_vs_best <- model_compare$AIC - min(model_compare$AIC)
write.csv(model_compare, file.path(outdir, "outcome_model_comparison.csv"), row.names = FALSE)

write_model_report(analysis$point_fits, dat, file.path(outdir, "model_summaries_and_diagnostics.txt"))
make_plots(dat, analysis$point_fits, analysis$results, outdir)

cat("\nAnalysis finished. Output folder: ", outdir, "\n", sep = "")
cat("Main result file: ", file.path(outdir, "mediation_results_bootstrap.csv"), "\n", sep = "")
cat("Model comparison file: ", file.path(outdir, "outcome_model_comparison.csv"), "\n", sep = "")
cat("Model report file: ", file.path(outdir, "model_summaries_and_diagnostics.txt"), "\n", sep = "")
cat("\nPreview of mediation results:\n")
print(head(analysis$results, 20))
