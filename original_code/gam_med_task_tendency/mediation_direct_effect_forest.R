#!/usr/bin/env Rscript

# ============================================================
# Grouped ACME forest plot for two mediators
# Fixed version: manually offsets y positions so CI lines remain horizontal.
#
# Purpose:
#   Draw grouped forest plots for ACME estimates from two mediators.
#   Each mediator is shown with different color, linetype, and point shape.
#   Point size represents absolute ACME magnitude.
#
# Required input:
#   Either:
#     1) Two mediation result CSV files, one per mediator; or
#     2) One combined CSV file with a mediator/group column.
#
# Expected columns in mediation result files:
#   model, contrast, age0, age1, effect, estimate, boot_se,
#   ci_low, ci_high, p_boot, n_boot_success
#
# Main output:
#   figure_grouped_ACME_forest_plot_two_mediators_FIXED.pdf
#   figure_grouped_ACME_forest_plot_two_mediators_FIXED.png
#   grouped_ACME_plotting_data.csv
# ============================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
})

# ============================================================
# 1. User configuration
# ============================================================

# -----------------------------
# Input mode
# -----------------------------
# Use "two_files" if you have two separate result files, one for each mediator.
# Use "combined" if you already have one result file containing a mediator column.
INPUT_MODE <- "two_files"
saveFIG <- FALSE

# -----------------------------
# Two-file input mode
# -----------------------------
# Modify the paths and mediator names below.
# Example:
# mediator_result_files <- c(
#   M1 = "local_linear_mediation_outputs_M1/local_linear_mediation_results.csv",
#   M2 = "local_linear_mediation_outputs_M2/local_linear_mediation_results.csv"
# )
mediator_result_files <- c(
  M1 = "/public/home/dingrui/tests/gam_mediation/reappraisal_success/mediation_results_bootstrap.csv",
  M2 = "/public/home/dingrui/tests/gam_mediation/relative_reappraisal_success/mediation_results_bootstrap.csv"
)

# -----------------------------
# Combined input mode
# -----------------------------
combined_result_file <- "combined_local_linear_mediation_results_twoM.csv"
mediator_column <- "mediator"

# -----------------------------
# Output directory
# -----------------------------
outdir <- "/public/home/dingrui/tests/gam_mediation"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 2. Plot selection
# ============================================================

# Which effect to plot. Usually "ACME_avg".
plot_effect <- "ACME_avg"

# Which model to plot:
#   "no_age_moderation"
#   "age_moderation"
#   "both"
plot_model <- "no_age_moderation"

# Which age contrast to plot
selected_contrasts <- c("8-12", "12-16", "16-19")

# If TRUE, only keep rows whose bootstrap succeeded at least this many times.
use_min_boot_success_filter <- FALSE
min_boot_success <- 500

# If TRUE, arrange contrasts by numeric age0 and age1.
# If FALSE, use the order found in the input files.
arrange_contrasts_by_age <- TRUE

# ============================================================
# 3. Visual customization: group appearance
# ============================================================

# Distance between the two mediator groups within one age contrast.
# Increase this if points/CI overlap too much.
group_offset <- 0.2

# Confidence interval line properties
ci_line_width <- 1.2
ci_line_alpha <- 0.90
ci_lineend <- "round"

# Zero reference line properties
zero_line_type <- "dashed"
zero_line_width <- 1
zero_line_alpha <- 0.8
zero_line_color <- "red"

# Point properties
point_alpha <- 0.95
point_stroke <- 0.85
point_size_range <- c(2.5, 7.0)

# Manual colors, linetypes, and shapes.
# Names must match mediator names in mediator_result_files or combined mediator column.
manual_colors <- c(
  M1 = "black",
  M2 = "black"
)

manual_linetypes <- c(
  M1 = "solid",
  M2 = "dashed"
)

# Shapes 21 and 24 support both color and fill.
manual_shapes <- c(
  M1 = 22,
  M2 = 22
)

# ============================================================
# 4. Visual customization: axes, fonts, ticks, figure size
# ============================================================

# Fonts and text sizes
base_family <- "DejaVu Sans"
base_size <- 12
plot_title_size <- 14
plot_subtitle_size <- 11
axis_title_size <- 12
axis_text_size <- 10.5
legend_text_size <- 10
legend_title_size <- 10.5
axis_text_x_color <- "black"
axis_text_y_color <- "black"
axis_title_x_color <- "black"
axis_title_y_color <- "black"

# Tick and axis line properties
axis_tick_length_pt <- 3.5
axis_tick_width <- 0.45
axis_line_width <- 0.45

# X-axis control. Set to NULL for automatic scaling.
# Example:
# x_limits <- c(-0.08, 0.09)
# x_breaks <- seq(-0.08, 0.08, by = 0.04)
x_limits <- NULL
x_breaks <- seq(-0.06, 0.09, by = 0.03)

# Additional y-axis expansion for top/bottom margins.
y_expand_mult <- c(0.08, 0.08)

# Legend control
legend_position <- "bottom"
legend_box <- "horizontal"

# Figure size
figure_width <- 3
figure_height <- 3
figure_units <- "in"
figure_dpi <- 300

# Output filenames
output_pdf <- file.path(outdir, "figure_grouped_ACME_forest_plot_two_mediators_FIXED.pdf")
output_png <- file.path(outdir, "figure_grouped_ACME_forest_plot_two_mediators_FIXED.png")
output_plot_data_csv <- file.path(outdir, "grouped_ACME_plotting_data.csv")

# ============================================================
# 5. Helper functions
# ============================================================

read_two_mediator_results <- function(file_vec) {
  if (length(file_vec) != 2) {
    stop("mediator_result_files must contain exactly two files.")
  }
  
  missing_files <- file_vec[!file.exists(file_vec)]
  if (length(missing_files) > 0) {
    stop(
      "The following input files do not exist:\n",
      paste(missing_files, collapse = "\n")
    )
  }
  
  out <- lapply(names(file_vec), function(med_name) {
    readr::read_csv(file_vec[[med_name]], show_col_types = FALSE) %>%
      dplyr::mutate(mediator = med_name)
  })
  
  dplyr::bind_rows(out)
}

read_combined_results <- function(file_path, mediator_col) {
  if (!file.exists(file_path)) {
    stop("combined_result_file does not exist: ", file_path)
  }
  
  dat <- readr::read_csv(file_path, show_col_types = FALSE)
  
  if (!mediator_col %in% names(dat)) {
    stop("The specified mediator_column is not present in combined_result_file: ", mediator_col)
  }
  
  if (mediator_col != "mediator") {
    dat <- dat %>% dplyr::rename(mediator = dplyr::all_of(mediator_col))
  }
  
  dat
}

check_required_columns <- function(dat) {
  required_cols <- c(
    "mediator", "model", "contrast", "age0", "age1", "effect",
    "estimate", "ci_low", "ci_high"
  )
  
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    stop(
      "The input data are missing required columns:\n",
      paste(missing_cols, collapse = ", ")
    )
  }
}

complete_manual_values <- function(groups, values, default_values, value_name) {
  missing_groups <- setdiff(groups, names(values))
  if (length(missing_groups) > 0) {
    n_missing <- length(missing_groups)
    if (length(default_values) < n_missing) {
      stop("Not enough default values for ", value_name, ".")
    }
    add_values <- default_values[seq_len(n_missing)]
    names(add_values) <- missing_groups
    values <- c(values, add_values)
  }
  values[groups]
}

# ============================================================
# 6. Read and validate input
# ============================================================

if (INPUT_MODE == "two_files") {
  all_results_twoM <- read_two_mediator_results(mediator_result_files)
} else if (INPUT_MODE == "combined") {
  all_results_twoM <- read_combined_results(combined_result_file, mediator_column)
} else {
  stop("INPUT_MODE must be either 'two_files' or 'combined'.")
}

check_required_columns(all_results_twoM)

# Coerce important columns to expected types.
all_results_twoM <- all_results_twoM %>%
  mutate(
    mediator = as.character(mediator),
    model = as.character(model),
    effect = as.character(effect),
    age0 = as.numeric(age0),
    age1 = as.numeric(age1),
    estimate = as.numeric(estimate),
    ci_low = as.numeric(ci_low),
    ci_high = as.numeric(ci_high)
  )

mediator_groups <- sort(unique(all_results_twoM$mediator))
if (length(mediator_groups) != 2) {
  stop(
    "This script expects exactly two mediator groups. Found: ",
    paste(mediator_groups, collapse = ", ")
  )
}

# Ensure manual aesthetics contain the two actual mediator groups.
manual_colors <- complete_manual_values(
  mediator_groups,
  manual_colors,
  default_values = c("#1B6CA8", "#D95F02", "#4D9221", "#7570B3"),
  value_name = "manual_colors"
)

manual_linetypes <- complete_manual_values(
  mediator_groups,
  manual_linetypes,
  default_values = c("solid", "twodash", "dotted", "dotdash"),
  value_name = "manual_linetypes"
)

manual_shapes <- complete_manual_values(
  mediator_groups,
  manual_shapes,
  default_values = c(21, 24, 22, 23),
  value_name = "manual_shapes"
)

# ============================================================
# 7. Prepare plotting data
# ============================================================

plot_dat <- all_results_twoM %>%
  filter(effect == plot_effect) %>%
  mutate(
    contrast_label = paste0(age0, "-", age1),
    mediator_group = factor(mediator),
    abs_acme = abs(estimate),
    ci_excludes_zero = ci_low > 0 | ci_high < 0
  ) %>%
  filter(contrast_label %in% selected_contrasts)

if (plot_model != "both") {
  plot_dat <- plot_dat %>% filter(model == plot_model)
}

if (use_min_boot_success_filter && "n_boot_success" %in% names(plot_dat)) {
  plot_dat <- plot_dat %>% filter(n_boot_success >= min_boot_success)
}

if (nrow(plot_dat) == 0) {
  stop("No rows remain after filtering. Check plot_effect and plot_model.")
}

# Keep only complete plotting rows.
plot_dat <- plot_dat %>%
  filter(
    is.finite(estimate),
    is.finite(ci_low),
    is.finite(ci_high),
    is.finite(age0),
    is.finite(age1)
  ) %>%
  mutate(
    contrast_label = paste0(age0, "-", age1),
    mediator_group = factor(mediator, levels = mediator_groups),
    abs_acme = abs(estimate),
    ci_excludes_zero = ci_low > 0 | ci_high < 0
  )

if (nrow(plot_dat) == 0) {
  stop("No complete rows remain after removing missing or non-finite values.")
}

# Stable age contrast order.
if (arrange_contrasts_by_age) {
  contrast_levels <- plot_dat %>%
    distinct(age0, age1, contrast_label) %>%
    arrange(age0, age1) %>%
    pull(contrast_label)
} else {
  contrast_levels <- unique(plot_dat$contrast_label)
}

plot_dat <- plot_dat %>%
  mutate(
    contrast_label = factor(contrast_label, levels = rev(contrast_levels)),
    y_base = as.numeric(contrast_label)
  )

# Manual y offsets. This fixes the slanted CI problem caused by position_dodge on geom_segment.
offset_tbl <- data.frame(
  mediator_group = factor(mediator_groups, levels = mediator_groups),
  y_offset = c(-group_offset, group_offset)
)

plot_dat <- plot_dat %>%
  left_join(offset_tbl, by = "mediator_group") %>%
  mutate(y_plot = y_base + y_offset)

# Save plotting data for reproducibility.
readr::write_csv(plot_dat, output_plot_data_csv)

# ============================================================
# 8. Build fixed grouped forest plot
# ============================================================

p_grouped_acme <- ggplot(plot_dat) +
  geom_vline(
    xintercept = 0,
    linetype = zero_line_type,
    linewidth = zero_line_width,
    alpha = zero_line_alpha,
    color = zero_line_color
  ) +
  geom_segment(
    aes(
      x = ci_low,
      xend = ci_high,
      y = y_plot,
      yend = y_plot,
      color = mediator_group,
      linetype = mediator_group
    ),
    linewidth = ci_line_width,
    alpha = ci_line_alpha,
    lineend = ci_lineend
  ) +
  geom_point(
    aes(
      x = estimate,
      y = y_plot,
      color = mediator_group,
      fill = mediator_group,
      shape = mediator_group,
      size = abs_acme
    ),
    alpha = point_alpha,
    stroke = point_stroke
  ) +
  scale_y_continuous(
    breaks = seq_along(levels(plot_dat$contrast_label)),
    labels = levels(plot_dat$contrast_label),
    expand = expansion(mult = y_expand_mult)
  ) +
  scale_color_manual(values = manual_colors) +
  scale_fill_manual(values = manual_colors) +
  scale_linetype_manual(values = manual_linetypes) +
  scale_shape_manual(values = manual_shapes) +
  scale_size_continuous(
    range = point_size_range,
    name = paste0("|", plot_effect, "|")
  ) +
  scale_x_continuous(breaks = x_breaks) +
  coord_cartesian(xlim = x_limits, clip = "off") +
  labs(
    x = paste0("GAM-based ", plot_effect, ", bootstrap 95% CI"),
    y = "Age contrast",
    color = "Mediator",
    fill = "Mediator",
    linetype = "Mediator",
    shape = "Mediator",
    title = "GAM mediation effects",
    subtitle = "Point size represents absolute ACME magnitude"
  ) +
  theme_classic(
    base_size = base_size,
    base_family = base_family
  ) +
  theme(
    plot.title = element_text(size = plot_title_size, face = "bold"),
    plot.subtitle = element_text(size = plot_subtitle_size),
    
    axis.title.x = element_text(
      size = axis_title_size,
      color = axis_title_x_color,
      margin = margin(t = 16)
    ),
    axis.title.y = element_text(
      size = axis_title_size,
      color = axis_title_y_color,
      margin = margin(r = 16)
    ),
    
    axis.text.x = element_text(
      size = axis_text_size,
      color = axis_text_x_color
    ),
    axis.text.y = element_text(
      size = axis_text_size,
      color = axis_text_y_color
    ),
    
    axis.ticks.length = grid::unit(axis_tick_length_pt, "pt"),
    axis.ticks = element_line(linewidth = axis_tick_width),
    axis.line = element_line(linewidth = axis_line_width),
    
    legend.title = element_text(size = legend_title_size),
    legend.text = element_text(size = legend_text_size),
    legend.position = "bottom",
    plot.title.position = "plot"
  )

# If plotting both models, facet them.
if (plot_model == "both") {
  p_grouped_acme <- p_grouped_acme +
    facet_wrap(~ model, nrow = 1, scales = "free_x")
}

# ============================================================
# 9. Save figure
# ============================================================

print(p_grouped_acme)

if (saveFIG){
  ggsave(
    filename = output_pdf,
    plot = p_grouped_acme,
    width = figure_width,
    height = figure_height,
    units = figure_units
  )
  
  ggsave(
    filename = output_png,
    plot = p_grouped_acme,
    width = figure_width,
    height = figure_height,
    units = figure_units,
    dpi = figure_dpi
  )
}

cat("\nGrouped ACME forest plot completed.\n")
if (saveFIG){
  cat("Plotting data saved to: ", output_plot_data_csv, "\n", sep = "")
  cat("PDF saved to: ", output_pdf, "\n", sep = "")
  cat("PNG saved to: ", output_png, "\n", sep = "")
}
