############################################################
# Local age-window traditional mediation analysis
# ----------------------------------------------------------
# Purpose:
#   Directly test age -> M -> outcome mediation within each
#   predefined age segment using ordinary least-squares linear
#   mediation models, without GAM smooth terms.
#
# Two outcome specifications are fitted in each age segment:
#   1) no_age_moderation:
#        M ~ age_c + covariates
#        Y ~ age_c + M + covariates
#
#   2) age_moderation:
#        M ~ age_c + covariates
#        Y ~ age_c * M + covariates
#
# Here age_c is centered at the midpoint of the segment, so the
# main effect of M in the interaction model is interpretable as
# the M -> outcome association at the segment midpoint.
#
# Main output:
#   local_linear_mediation_outputs/local_linear_mediation_results.csv
#   local_linear_mediation_outputs/local_linear_path_coefficients.csv
#   local_linear_mediation_outputs/local_linear_model_summaries.txt
#   local_linear_mediation_outputs/figure_local_ACME_forest_plot.pdf
#   local_linear_mediation_outputs/figure_local_effect_decomposition_forest_plot.pdf
#   local_linear_mediation_outputs/figure_local_ACME_model_comparison.pdf
#
# Additional segment-level outputs in this modified version:
#   local_linear_mediation_outputs/segment_data_tables/
#       <segment>_original_data.csv
#       <segment>_analysis_data.csv
#   local_linear_mediation_outputs/segment_pairwise_scatterplots/
#       <segment>_pairwise_scatterplots.pdf
#       <segment>_pairwise_scatterplots.png
#   local_linear_mediation_outputs/local_linear_segment_data_index.csv
############################################################

############################
# 0. User settings
############################

# Set to FALSE for your own CSV. If TRUE and input_csv does not exist,
# a simulated dataset will be generated.
RUN_SIMULATION_IF_NEEDED <- FALSE

input_csv <- "/public/home/dingrui/fmri_analysis/data/beh/er_profiles_new.csv"
outdir <- "/public/home/dingrui/tests/linear_mediation"

# Output segment-specific data tables and pairwise scatterplots.
# Each age segment will get two CSV tables:
#   1) original_data: original columns from your input CSV, filtered to the segment
#   2) analysis_data: internal modeling variables used by this script, including age_c
OUTPUT_SEGMENT_TABLES <- TRUE
OUTPUT_SEGMENT_SCATTERPLOTS <- TRUE

# Column names in your CSV
id_var       <- "sub_id"
age_var      <- "age"
gender_var   <- "sex"
mediator_var <- "relative_reappraisal_success"
outcome_var  <- "ERQ_CR"

# Covariates. Include gender here if you want to adjust for gender.
# All character/factor covariates will be treated as factors.
covariates <- c("sex")

# Age contrasts and local age windows.
# age0/age1 define the mediation contrast.
# lower/upper define which participants are used to fit the local linear model.
# By default lower=age0 and upper=age1, meaning the model is fitted only in that interval.
age_segments <- data.frame(
  label = c("8_to_12", "12_to_16", "16_to_19", "8_to_19", "8_to_11", "11_to_19"),
  age0  = c(8, 12, 16, 8, 8, 11),
  age1  = c(12, 16, 19, 19, 11, 19),
  lower = c(8, 12, 16, 8, 8, 11),
  upper = c(12, 16, 19, 19, 11, 19)
)

# Optional buffer around each fitting window. For example, 0.5 means
# contrast 8->12 is fitted using ages 7.5 to 12.5.
window_buffer <- 0

# Minimum sample size required to run a segment-specific model.
min_n <- 40

# Bootstrap settings for mediation::mediate().
BOOT_SIMS <- 1000
BOOT_CI_TYPE <- "perc"
SEED <- 20260621

# Fit both traditional models:
#   no_age_moderation = Y ~ age_c + M + covariates
#   age_moderation    = Y ~ age_c * M + covariates
fit_model_types <- c("no_age_moderation", "age_moderation")

############################
# 1. Packages
############################

required_packages <- c("mediation", "dplyr", "readr", "ggplot2", "forcats", "tibble", "purrr", "stringr", "tidyr")

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_if_missing(required_packages)

suppressPackageStartupMessages({
  library(mediation)
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(forcats)
  library(tibble)
  library(purrr)
  library(stringr)
})

set.seed(SEED)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
segment_table_dir <- file.path(outdir, "segment_data_tables")
segment_plot_dir <- file.path(outdir, "segment_pairwise_scatterplots")
if (OUTPUT_SEGMENT_TABLES) {
  dir.create(segment_table_dir, showWarnings = FALSE, recursive = TRUE)
}
if (OUTPUT_SEGMENT_SCATTERPLOTS) {
  dir.create(segment_plot_dir, showWarnings = FALSE, recursive = TRUE)
}

############################
# 2. Optional simulated data
############################

simulate_demo_data <- function(n = 800, seed = 20260621) {
  set.seed(seed)
  sub_id <- sprintf("sub_%04d", seq_len(n))
  age <- runif(n, 6, 19)
  gender <- factor(sample(c("female", "male"), n, replace = TRUE))
  gender_num <- ifelse(gender == "male", 1, 0)

  # Nonlinear age -> M trajectory: rises, then falls after early adolescence.
  M_mean <- 0.55 * sin((age - 6) / 13 * pi * 1.3) - 0.05 * (age - 13)
  M <- M_mean + 0.18 * gender_num + rnorm(n, 0, 0.75)

  # Outcome: linear M -> outcome, plus weak age-by-M moderation.
  Y <- 0.03 * (age - 12) +
    0.32 * M +
    0.035 * (age - 12) * M +
    0.12 * gender_num +
    rnorm(n, 0, 1.00)

  data.frame(sub_id = sub_id, age = age, gender = gender, M = M, outcome = Y)
}

if (!file.exists(input_csv)) {
  if (RUN_SIMULATION_IF_NEEDED) {
    message("input_csv not found. Generating simulated data: ", input_csv)
    demo_dat <- simulate_demo_data()
    write_csv(demo_dat, input_csv)
  } else {
    stop("input_csv does not exist: ", input_csv)
  }
}

############################
# 3. Load and prepare data
############################

raw_dat <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(.row_id = dplyr::row_number())

required_cols <- unique(c(id_var, age_var, mediator_var, outcome_var, covariates))
missing_cols <- setdiff(required_cols, names(raw_dat))
if (length(missing_cols) > 0) {
  stop("The following required columns are missing from the CSV: ", paste(missing_cols, collapse = ", "))
}

# Use clean internal variable names for modeling.
dat <- raw_dat %>%
  mutate(
    .row_id = .data$.row_id,
    .sub_id = .data[[id_var]],
    age = as.numeric(.data[[age_var]]),
    M = as.numeric(.data[[mediator_var]]),
    Y = as.numeric(.data[[outcome_var]])
  )

# Copy covariates to syntactically valid internal names.
covariate_internal <- make.names(covariates, unique = TRUE)
for (i in seq_along(covariates)) {
  dat[[covariate_internal[i]]] <- raw_dat[[covariates[i]]]
  if (is.character(dat[[covariate_internal[i]]]) || is.factor(dat[[covariate_internal[i]]])) {
    dat[[covariate_internal[i]]] <- factor(dat[[covariate_internal[i]]])
  }
}

analysis_vars <- c(".row_id", ".sub_id", "age", "M", "Y", covariate_internal)
dat <- dat[, analysis_vars, drop = FALSE]
dat <- dat[is.finite(dat$age) & is.finite(dat$M) & is.finite(dat$Y), , drop = FALSE]
dat <- tidyr::drop_na(dat)

if (nrow(dat) < min_n) {
  stop("Too few complete observations after preprocessing: n = ", nrow(dat))
}

message("Complete-case sample size: ", nrow(dat))
message("Age range: ", round(min(dat$age), 3), " to ", round(max(dat$age), 3))

############################
# 4. Helper functions
############################

make_formula <- function(lhs, rhs_terms) {
  rhs_terms <- rhs_terms[nzchar(rhs_terms)]
  as.formula(paste(lhs, "~", paste(rhs_terms, collapse = " + ")))
}

# mediation::mediate() may re-evaluate the original lm() call during
# bootstrap. If an lm was fitted inside a function using a local object
# such as m_formula, mediate() can fail with:
#   object 'm_formula' not found
# This helper makes the fitted lm call self-contained by storing the
# actual formula object and by pointing the call to a temporary data
# object that is accessible during mediate().
fit_lm_for_mediate <- function(formula_obj, data_obj, data_name) {
  assign(data_name, data_obj, envir = .GlobalEnv)
  fit <- lm(formula_obj, data = get(data_name, envir = .GlobalEnv))
  fit$call$formula <- formula_obj
  fit$call$data <- as.name(data_name)
  fit
}

extract_lm_coef <- function(fit, term) {
  cf <- summary(fit)$coefficients
  if (!term %in% rownames(cf)) {
    return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_))
  }
  tibble(
    estimate = unname(cf[term, "Estimate"]),
    se = unname(cf[term, "Std. Error"]),
    t = unname(cf[term, "t value"]),
    p = unname(cf[term, "Pr(>|t|)"])
  )
}

extract_mediate_effects <- function(med_obj, segment_row, model_type, n_segment, age_mid, window_lower, window_upper) {
  effect_map <- list(
    ACME_at_age0 = list(est = "d0",       ci = "d0.ci",     p = "d0.p"),
    ACME_at_age1 = list(est = "d1",       ci = "d1.ci",     p = "d1.p"),
    ACME_avg     = list(est = "d.avg",    ci = "d.avg.ci",  p = "d.avg.p"),
    ADE_Ma0      = list(est = "z0",       ci = "z0.ci",     p = "z0.p"),
    ADE_Ma1      = list(est = "z1",       ci = "z1.ci",     p = "z1.p"),
    ADE_avg      = list(est = "z.avg",    ci = "z.avg.ci",  p = "z.avg.p"),
    Total_effect = list(est = "tau.coef", ci = "tau.ci",    p = "tau.p"),
    Prop_med_avg = list(est = "n.avg",    ci = "n.avg.ci",  p = "n.avg.p")
  )

  purrr::imap_dfr(effect_map, function(keys, effect_name) {
    est <- med_obj[[keys$est]]
    ci  <- med_obj[[keys$ci]]
    p   <- med_obj[[keys$p]]

    tibble(
      model = model_type,
      segment = as.character(segment_row$label),
      age0 = as.numeric(segment_row$age0),
      age1 = as.numeric(segment_row$age1),
      age_mid = age_mid,
      window_lower = window_lower,
      window_upper = window_upper,
      n = n_segment,
      effect = effect_name,
      estimate = as.numeric(est),
      ci_low = as.numeric(ci[1]),
      ci_high = as.numeric(ci[2]),
      p_value = as.numeric(p),
      boot_sims = BOOT_SIMS
    )
  })
}



sanitize_filename <- function(x) {
  x <- as.character(x)
  x <- stringr::str_replace_all(x, "[^A-Za-z0-9_\\-]+", "_")
  x <- stringr::str_replace_all(x, "_+", "_")
  x <- stringr::str_replace_all(x, "^_|_$", "")
  ifelse(nchar(x) == 0, "segment", x)
}

make_segment_files <- function(seg, subdat, age_mid, window_lower, window_upper) {
  label <- as.character(seg$label)
  safe_label <- sanitize_filename(label)
  n_segment <- nrow(subdat)

  original_csv <- file.path(segment_table_dir, paste0(safe_label, "_original_data.csv"))
  analysis_csv <- file.path(segment_table_dir, paste0(safe_label, "_analysis_data.csv"))
  scatter_pdf <- file.path(segment_plot_dir, paste0(safe_label, "_pairwise_scatterplots.pdf"))
  scatter_png <- file.path(segment_plot_dir, paste0(safe_label, "_pairwise_scatterplots.png"))

  if (OUTPUT_SEGMENT_TABLES) {
    original_segment <- raw_dat %>%
      filter(.row_id %in% subdat$.row_id) %>%
      mutate(
        .segment = label,
        .contrast_age0 = as.numeric(seg$age0),
        .contrast_age1 = as.numeric(seg$age1),
        .window_lower = window_lower,
        .window_upper = window_upper,
        .age_mid = age_mid
      ) %>%
      arrange(.row_id)

    analysis_segment <- subdat %>%
      mutate(
        segment = label,
        contrast_age0 = as.numeric(seg$age0),
        contrast_age1 = as.numeric(seg$age1),
        window_lower = window_lower,
        window_upper = window_upper,
        age_mid = age_mid
      ) %>%
      arrange(age, .sub_id)

    write_csv(original_segment, original_csv)
    write_csv(analysis_segment, analysis_csv)
  } else {
    original_csv <- NA_character_
    analysis_csv <- NA_character_
  }

  scatter_status <- "not_requested"
  scatter_error <- NA_character_

  if (OUTPUT_SEGMENT_SCATTERPLOTS) {
    if (n_segment < 2) {
      scatter_status <- "skipped_n_less_than_2"
      scatter_pdf <- NA_character_
      scatter_png <- NA_character_
    } else {
      pair_dat <- bind_rows(
        subdat %>% transmute(pair = paste0(age_var, " vs ", mediator_var), x = age, y = M),
        subdat %>% transmute(pair = paste0(age_var, " vs ", outcome_var), x = age, y = Y),
        subdat %>% transmute(pair = paste0(mediator_var, " vs ", outcome_var), x = M, y = Y)
      ) %>%
        mutate(pair = factor(
          pair,
          levels = c(
            paste0(age_var, " vs ", mediator_var),
            paste0(age_var, " vs ", outcome_var),
            paste0(mediator_var, " vs ", outcome_var)
          )
        ))

      p_pairs <- ggplot(pair_dat, aes(x = x, y = y)) +
        geom_point(alpha = 0.65, size = 1.6) +
        geom_smooth(method = "lm", se = TRUE, linewidth = 0.6) +
        facet_wrap(~ pair, scales = "free", nrow = 1) +
        labs(
          x = "X variable value",
          y = "Y variable value",
          title = paste0("Pairwise scatterplots for age segment: ", label),
          subtitle = paste0(
            "Fitting window: ", round(window_lower, 3), "-", round(window_upper, 3),
            "; n = ", n_segment
          )
        ) +
        theme_classic(base_size = 11) +
        theme(
          strip.background = element_blank(),
          plot.title.position = "plot"
        )

      plot_try <- tryCatch({
        ggsave(scatter_pdf, p_pairs, width = 10.5, height = 3.8)
        ggsave(scatter_png, p_pairs, width = 10.5, height = 3.8, dpi = 300)
        "saved"
      }, error = function(e) {
        scatter_pdf <<- NA_character_
        scatter_png <<- NA_character_
        scatter_error <<- conditionMessage(e)
        "failed"
      })
      scatter_status <- plot_try
    }
  } else {
    scatter_pdf <- NA_character_
    scatter_png <- NA_character_
  }

  tibble(
    segment = label,
    age0 = as.numeric(seg$age0),
    age1 = as.numeric(seg$age1),
    window_lower = window_lower,
    window_upper = window_upper,
    age_mid = age_mid,
    n = n_segment,
    original_data_csv = original_csv,
    analysis_data_csv = analysis_csv,
    pairwise_scatter_pdf = scatter_pdf,
    pairwise_scatter_png = scatter_png,
    scatter_status = scatter_status,
    scatter_error = scatter_error
  )
}


safe_mediate <- function(model_m, model_y, control_value, treat_value) {
  mediation::mediate(
    model.m = model_m,
    model.y = model_y,
    treat = "age_c",
    mediator = "M",
    control.value = control_value,
    treat.value = treat_value,
    sims = BOOT_SIMS,
    boot = TRUE,
    boot.ci.type = BOOT_CI_TYPE,
    long = TRUE,
    dropobs = TRUE
  )
}

fit_one_segment <- function(seg) {
  label <- as.character(seg$label)
  age0 <- as.numeric(seg$age0)
  age1 <- as.numeric(seg$age1)
  lower <- as.numeric(seg$lower) - window_buffer
  upper <- as.numeric(seg$upper) + window_buffer
  age_mid <- mean(c(age0, age1))

  subdat <- dat %>%
    filter(age >= lower, age <= upper) %>%
    mutate(age_c = age - age_mid)

  n_segment <- nrow(subdat)

  # Save the data actually used for this segment and create diagnostic scatterplots
  # before applying min_n. This means every user-defined segment receives a data table,
  # even if it is later skipped for mediation because the sample size is too small.
  data_index <- make_segment_files(
    seg = seg,
    subdat = subdat,
    age_mid = age_mid,
    window_lower = lower,
    window_upper = upper
  )

  # Create a unique temporary global data object for mediation bootstrap
  # re-fitting. This avoids lm call objects that point to local variables.
  tmp_data_name <- paste0(".local_mediation_data_", gsub("[^A-Za-z0-9_]", "_", label))

  if (n_segment < min_n) {
    warning("Skipping segment ", label, ": n = ", n_segment, " < min_n = ", min_n)
    return(list(
      results = tibble(),
      paths = tibble(),
      summaries = paste0("\n\n===== Segment ", label, " skipped: n = ", n_segment, " < ", min_n, " =====\n"),
      failures = tibble(segment = label, model = NA_character_, error = paste0("n < min_n: ", n_segment))
    ))
  }

  # Formulas.
  cov_terms <- covariate_internal
  m_formula <- make_formula("M", c("age_c", cov_terms))
  y_formula_no <- make_formula("Y", c("age_c", "M", cov_terms))
  y_formula_int <- make_formula("Y", c("age_c * M", cov_terms))
  total_formula <- make_formula("Y", c("age_c", cov_terms))

  # Fit mediator and total-effect model once per segment.
  # Use fit_lm_for_mediate() instead of bare lm() so mediation::mediate()
  # can safely re-evaluate the model call during bootstrap.
  m_fit <- fit_lm_for_mediate(m_formula, subdat, tmp_data_name)
  total_fit <- fit_lm_for_mediate(total_formula, subdat, tmp_data_name)

  segment_results <- list()
  path_results <- list()
  failure_results <- list()

  summary_text <- paste0(
    "\n\n============================================================\n",
    "Segment: ", label, "\n",
    "Contrast: ", age0, " -> ", age1, "\n",
    "Fitting window: [", lower, ", ", upper, "]\n",
    "N: ", n_segment, "\n",
    "Age centered at: ", age_mid, "\n",
    "============================================================\n\n",
    "Mediator model: ", deparse(m_formula), "\n",
    paste(capture.output(summary(m_fit)), collapse = "\n"),
    "\n\nTotal-effect model: ", deparse(total_formula), "\n",
    paste(capture.output(summary(total_fit)), collapse = "\n")
  )

  # Common path a and total c.
  a_coef <- extract_lm_coef(m_fit, "age_c")
  c_coef <- extract_lm_coef(total_fit, "age_c")

  for (model_type in fit_model_types) {
    y_formula <- if (model_type == "no_age_moderation") y_formula_no else y_formula_int
    y_fit <- fit_lm_for_mediate(y_formula, subdat, tmp_data_name)

    summary_text <- paste0(
      summary_text,
      "\n\nOutcome model [", model_type, "]: ", deparse(y_formula), "\n",
      paste(capture.output(summary(y_fit)), collapse = "\n"),
      "\n"
    )

    control_value <- age0 - age_mid
    treat_value <- age1 - age_mid

    med_obj <- tryCatch(
      safe_mediate(m_fit, y_fit, control_value, treat_value),
      error = function(e) e
    )

    if (inherits(med_obj, "error")) {
      failure_results[[model_type]] <- tibble(
        segment = label,
        model = model_type,
        error = conditionMessage(med_obj)
      )
      next
    }

    segment_results[[model_type]] <- extract_mediate_effects(
      med_obj = med_obj,
      segment_row = seg,
      model_type = model_type,
      n_segment = n_segment,
      age_mid = age_mid,
      window_lower = lower,
      window_upper = upper
    )

    # Path coefficients.
    b_mid <- extract_lm_coef(y_fit, "M")
    cprime <- extract_lm_coef(y_fit, "age_c")
    int_coef <- extract_lm_coef(y_fit, "age_c:M")
    if (is.na(int_coef$estimate)) {
      int_coef <- extract_lm_coef(y_fit, "M:age_c")
    }

    path_results[[model_type]] <- tibble(
      model = model_type,
      segment = label,
      age0 = age0,
      age1 = age1,
      age_mid = age_mid,
      window_lower = lower,
      window_upper = upper,
      n = n_segment,
      a_age_to_M = a_coef$estimate,
      a_se = a_coef$se,
      a_p = a_coef$p,
      b_M_to_Y_at_mid_age = b_mid$estimate,
      b_se = b_mid$se,
      b_p = b_mid$p,
      cprime_direct_age_to_Y = cprime$estimate,
      cprime_se = cprime$se,
      cprime_p = cprime$p,
      c_total_age_to_Y = c_coef$estimate,
      c_total_se = c_coef$se,
      c_total_p = c_coef$p,
      age_M_interaction = int_coef$estimate,
      age_M_interaction_se = int_coef$se,
      age_M_interaction_p = int_coef$p,
      product_ab_for_age_contrast = a_coef$estimate * b_mid$estimate * (age1 - age0)
    )
  }

  list(
    results = bind_rows(segment_results),
    paths = bind_rows(path_results),
    summaries = summary_text,
    failures = bind_rows(failure_results),
    data_index = data_index
  )
}

############################
# 5. Run local mediation analyses
############################

message("Running local traditional mediation analyses...")
message("Bootstrap sims per model: ", BOOT_SIMS)

segment_outputs <- purrr::pmap(
  age_segments,
  function(label, age0, age1, lower, upper) {
    fit_one_segment(data.frame(label = label, age0 = age0, age1 = age1, lower = lower, upper = upper))
  }
)

all_results <- bind_rows(purrr::map(segment_outputs, "results"))
all_paths <- bind_rows(purrr::map(segment_outputs, "paths"))
all_failures <- bind_rows(purrr::map(segment_outputs, "failures"))
all_data_index <- bind_rows(purrr::map(segment_outputs, "data_index"))
all_summaries <- paste(purrr::map_chr(segment_outputs, "summaries"), collapse = "\n")

write_csv(all_results, file.path(outdir, "local_linear_mediation_results.csv"))
write_csv(all_paths, file.path(outdir, "local_linear_path_coefficients.csv"))
writeLines(all_summaries, con = file.path(outdir, "local_linear_model_summaries.txt"))
write_csv(all_data_index, file.path(outdir, "local_linear_segment_data_index.csv"))

if (nrow(all_failures) > 0) {
  write_csv(all_failures, file.path(outdir, "local_linear_model_failures.csv"))
}

message("Saved mediation results to: ", file.path(outdir, "local_linear_mediation_results.csv"))
message("Saved path coefficients to: ", file.path(outdir, "local_linear_path_coefficients.csv"))
message("Saved model summaries to: ", file.path(outdir, "local_linear_model_summaries.txt"))
message("Saved segment data index to: ", file.path(outdir, "local_linear_segment_data_index.csv"))
if (OUTPUT_SEGMENT_TABLES) {
  message("Saved segment-level data tables to: ", segment_table_dir)
}
if (OUTPUT_SEGMENT_SCATTERPLOTS) {
  message("Saved segment-level pairwise scatterplots to: ", segment_plot_dir)
}

############################
# 6. Plots
############################

if (nrow(all_results) > 0) {
  # Main ACME forest plot.
  acme_plot_dat <- all_results %>%
    filter(effect == "ACME_avg") %>%
    mutate(
      contrast_label = paste0(age0, "-", age1),
      contrast_label = forcats::fct_reorder(contrast_label, age0 + age1 / 100, .desc = TRUE),
      ci_excludes_zero = ci_low > 0 | ci_high < 0,
      sig_label = ifelse(ci_excludes_zero, "95% CI excludes 0", "95% CI includes 0"),
      model_label = recode(
        model,
        no_age_moderation = "No age moderation",
        age_moderation = "Age × M moderation"
      )
    )

  p_acme <- ggplot(acme_plot_dat, aes(y = contrast_label)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_segment(aes(x = ci_low, xend = ci_high, yend = contrast_label), linewidth = 0.65) +
    geom_point(aes(x = estimate, shape = sig_label), size = 2.5) +
    facet_wrap(~ model_label, nrow = 1) +
    labs(
      x = "Local linear ACME_avg, bootstrap 95% CI",
      y = "Age contrast",
      shape = NULL,
      title = "Local age-window traditional mediation analysis",
      subtitle = "Each age contrast is fitted using only observations in the corresponding local age window"
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.background = element_blank(),
      plot.title.position = "plot"
    )

  ggsave(
    filename = file.path(outdir, "figure_local_ACME_forest_plot.pdf"),
    plot = p_acme,
    width = 9,
    height = 4.8
  )

  # Decomposition plot: ACME, ADE, total effect.
  decomp_plot_dat <- all_results %>%
    filter(effect %in% c("ACME_avg", "ADE_avg", "Total_effect")) %>%
    mutate(
      contrast_label = paste0(age0, "-", age1),
      contrast_label = forcats::fct_reorder(contrast_label, age0 + age1 / 100, .desc = TRUE),
      effect_label = recode(
        effect,
        ACME_avg = "Indirect effect\nACME",
        ADE_avg = "Direct effect\nADE",
        Total_effect = "Total effect"
      ),
      model_label = recode(
        model,
        no_age_moderation = "No age moderation",
        age_moderation = "Age × M moderation"
      ),
      ci_excludes_zero = ci_low > 0 | ci_high < 0,
      sig_label = ifelse(ci_excludes_zero, "95% CI excludes 0", "95% CI includes 0")
    )

  p_decomp <- ggplot(decomp_plot_dat, aes(y = contrast_label)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_segment(aes(x = ci_low, xend = ci_high, yend = contrast_label), linewidth = 0.6) +
    geom_point(aes(x = estimate, shape = sig_label), size = 2.2) +
    facet_grid(model_label ~ effect_label, scales = "free_x") +
    labs(
      x = "Estimated effect on outcome scale, bootstrap 95% CI",
      y = "Age contrast",
      shape = NULL,
      title = "Effect decomposition from local traditional mediation models"
    ) +
    theme_classic(base_size = 11) +
    theme(
      legend.position = "bottom",
      strip.background = element_blank(),
      plot.title.position = "plot"
    )

  ggsave(
    filename = file.path(outdir, "figure_local_effect_decomposition_forest_plot.pdf"),
    plot = p_decomp,
    width = 11,
    height = 6.5
  )

  # Direct comparison of ACME from no-moderation vs age-moderation models.
  p_compare <- ggplot(acme_plot_dat, aes(x = contrast_label, y = estimate, group = model_label)) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.12, position = position_dodge(width = 0.5)) +
    geom_point(aes(shape = model_label), size = 2.5, position = position_dodge(width = 0.5)) +
    coord_flip() +
    labs(
      x = "Age contrast",
      y = "ACME_avg, bootstrap 95% CI",
      shape = NULL,
      title = "Local traditional ACME: no moderation vs age × M moderation"
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title.position = "plot"
    )

  ggsave(
    filename = file.path(outdir, "figure_local_ACME_model_comparison.pdf"),
    plot = p_compare,
    width = 8,
    height = 5
  )
}

############################
# 7. Console summary
############################

# Clean temporary objects created for mediation bootstrap refitting.
rm(list = grep("^\\.local_mediation_data_", ls(envir = .GlobalEnv), value = TRUE), envir = .GlobalEnv)

message("\nDone.")
message("Main files:")
message("  - ", file.path(outdir, "local_linear_mediation_results.csv"))
message("  - ", file.path(outdir, "local_linear_path_coefficients.csv"))
message("  - ", file.path(outdir, "local_linear_model_summaries.txt"))
message("  - ", file.path(outdir, "figure_local_ACME_forest_plot.pdf"))
message("  - ", file.path(outdir, "figure_local_effect_decomposition_forest_plot.pdf"))
message("  - ", file.path(outdir, "figure_local_ACME_model_comparison.pdf"))
message("  - ", file.path(outdir, "local_linear_segment_data_index.csv"))
message("  - ", segment_table_dir)
message("  - ", segment_plot_dir)

# Suggested interpretation:
#   local_linear_mediation_results.csv:
#     Use ACME_avg as the main local indirect effect.
#     Use ADE_avg and Total_effect to decide whether the age-outcome association
#     is mainly direct or mediated.
#     Treat Prop_med_avg cautiously when Total_effect is small or nonsignificant.
#
#   local_linear_path_coefficients.csv:
#     a_age_to_M: local linear age -> M slope within the window.
#     b_M_to_Y_at_mid_age: local M -> Y association at the segment midpoint.
#     product_ab_for_age_contrast: conventional product-of-coefficients estimate
#       scaled to the age contrast. This is most interpretable in the
#       no_age_moderation model.
