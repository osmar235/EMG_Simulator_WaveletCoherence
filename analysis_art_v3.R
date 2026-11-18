# analysis_art_v3.R -- Version 3.3
#
# PRIMARY corrected analysis:
#   mixed ART with a random intercept for bilateral trial because the four band
#   values from one trial are repeated/correlated observations.
#
# SECONDARY manuscript-reproduction analysis:
#   fixed-effects ART matching the original published analysis structure.
#
# Required packages:
#   install.packages(c("ARTool", "dplyr", "readr"))
#
# Usage:
#   Rscript analysis_art_v3.R results/factorial_v3_3/factorial_v3_3_band_medians.csv

suppressPackageStartupMessages({
  library(ARTool)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
infile <- if (length(args) >= 1) args[[1]] else "results/factorial_v3_3/factorial_v3_3_band_medians.csv"
outdir <- file.path(dirname(infile), "ART_v3_3")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

d <- read_csv(infile, show_col_types = FALSE) %>%
  mutate(
    crosstalk = factor(crosstalk_R2L, levels = sort(unique(crosstalk_R2L))),
    mixture = factor(mixture, levels = c("100_0", "75_25", "50_50")),
    cortical_high = factor(cortical_high_hz, levels = c(30, 60)),
    band = factor(band, levels = c("5-13", "13-30", "30-60", "60-100")),
    trial_id = if ("trial_uid" %in% names(.)) factor(trial_uid) else
      interaction(cortical_high, mixture, crosstalk, trial, drop = TRUE)
  )

# -----------------------------------------------------------------------------
# 1. PRIMARY V3.3 corrected model: repeated band values within bilateral trial.
# -----------------------------------------------------------------------------
m_primary <- art(
  coherence_median ~ crosstalk * mixture * cortical_high * band + (1 | trial_id),
  data = d
)

capture.output(summary(m_primary), file = file.path(outdir, "01_PRIMARY_mixed_ART_summary.txt"))
write_csv(as.data.frame(anova(m_primary, response = "aligned")),
          file.path(outdir, "02_PRIMARY_alignment_diagnostics.csv"))
write_csv(as.data.frame(anova(m_primary)),
          file.path(outdir, "03_PRIMARY_mixed_ART_typeIII_ANOVA.csv"))

write_primary_contrasts <- function(term, filename) {
  z <- art.con(m_primary, term, method = "pairwise", adjust = "tukey")
  write_csv(as.data.frame(summary(z)), file.path(outdir, filename))
}

write_primary_contrasts("crosstalk", "10_PRIMARY_posthoc_crosstalk.csv")
write_primary_contrasts("mixture", "11_PRIMARY_posthoc_mixture.csv")
write_primary_contrasts("band", "12_PRIMARY_posthoc_band.csv")
write_primary_contrasts("cortical_high", "13_PRIMARY_posthoc_cortical_bandwidth.csv")
write_primary_contrasts("mixture:crosstalk", "20_PRIMARY_posthoc_mixture_x_crosstalk.csv")
write_primary_contrasts("crosstalk:band", "21_PRIMARY_posthoc_crosstalk_x_band.csv")
write_primary_contrasts("cortical_high:band:mixture", "22_PRIMARY_posthoc_bandwidth_x_band_x_mixture.csv")

# Descriptive summaries.
desc <- d %>%
  group_by(cortical_high, mixture, crosstalk, band) %>%
  summarise(
    n = n(),
    mean = mean(coherence_median, na.rm = TRUE),
    sd = sd(coherence_median, na.rm = TRUE),
    median = median(coherence_median, na.rm = TRUE),
    q1 = quantile(coherence_median, 0.25, na.rm = TRUE),
    q3 = quantile(coherence_median, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(desc, file.path(outdir, "30_cell_descriptives.csv"))

# -----------------------------------------------------------------------------
# 2. SECONDARY manuscript-reproduction model: original fixed-effects ART.
# -----------------------------------------------------------------------------
m_fixed <- art(coherence_median ~ crosstalk * mixture * cortical_high * band, data = d)
capture.output(summary(m_fixed), file = file.path(outdir, "40_REPRO_fixed_ART_summary.txt"))
write_csv(as.data.frame(anova(m_fixed)),
          file.path(outdir, "41_REPRO_fixed_ART_typeIII_ANOVA.csv"))

write_fixed_contrasts <- function(term, filename) {
  z <- art.con(m_fixed, term, method = "pairwise", adjust = "tukey")
  write_csv(as.data.frame(summary(z)), file.path(outdir, filename))
}
write_fixed_contrasts("mixture:crosstalk", "42_REPRO_posthoc_mixture_x_crosstalk.csv")
write_fixed_contrasts("crosstalk:band", "43_REPRO_posthoc_crosstalk_x_band.csv")
write_fixed_contrasts("cortical_high:band:mixture", "44_REPRO_posthoc_bandwidth_x_band_x_mixture.csv")

writeLines(c(
  "V3.3 analysis hierarchy:",
  "PRIMARY: mixed ART with (1|trial_id), acknowledging repeated frequency-band values from each bilateral trial.",
  "SECONDARY: fixed-effects ART retained only to reproduce/compare with the original manuscript analysis.",
  "Absolute coherence/null-floor interpretation should use trial_shuffle_null_v3.py outputs, not phase-randomized thresholds."
), file.path(outdir, "00_README_ANALYSIS.txt"))

cat("Saved V3.3 ART outputs to:", normalizePath(outdir), "\n")
