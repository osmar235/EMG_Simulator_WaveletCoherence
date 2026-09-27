# art_analysis.R - Aligned Rank Transform ANOVA of the factorial study.
#
# Factors: crosstalk (8 levels) x mixture (3) x cortical bandwidth (2) x band (4).
# Model 1 (recommended): mixed ART with a random intercept per simulated trial,
#          because the four band values of one trial are repeated measures.
# Model 2: fixed-effects ART (all observations treated as independent).
#
# Install once:  install.packages(c("ARTool", "lme4", "emmeans", "dplyr", "readr"))
# Usage:         Rscript paper_analysis/art_analysis.R results/factorial/factorial_band_medians.csv
# Output:        <folder>/ART/*.csv / *.txt

suppressPackageStartupMessages({
  library(ARTool); library(dplyr); library(readr)
})
# With ~720 trials the denominator df are very large; asymptotic (z) df for the
# mixed-model contrasts give the same inference much faster than Kenward-Roger.
emmeans::emm_options(lmer.df = "asymptotic")
t0 <- Sys.time()

args <- commandArgs(trailingOnly = TRUE)
infile <- if (length(args) >= 1) args[[1]] else "results/factorial/factorial_band_medians.csv"
outdir <- file.path(dirname(infile), "ART")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

d <- read_csv(infile, show_col_types = FALSE) %>%
  mutate(
    crosstalk     = factor(crosstalk_R2L, levels = sort(unique(crosstalk_R2L))),
    mixture       = factor(mixture, levels = c("100_0", "75_25", "50_50")),
    cortical_high = factor(cortical_high_hz, levels = c(30, 60)),
    band          = factor(band, levels = c("5-13", "13-30", "30-60", "60-100")),
    trial_id      = factor(trial_id)
  )

contrasts_for <- function(model, prefix) {
  for (term in c("crosstalk", "mixture", "band", "cortical_high",
                 "mixture:crosstalk", "crosstalk:band", "cortical_high:band:mixture")) {
    z <- art.con(model, term, method = "pairwise", adjust = "tukey")
    write_csv(as.data.frame(summary(z)),
              file.path(outdir, paste0(prefix, "_posthoc_", gsub(":", "_x_", term), ".csv")))
  }
}

# Model 1: mixed ART
m1 <- art(coherence_median ~ crosstalk * mixture * cortical_high * band + (1 | trial_id), data = d)
capture.output(summary(m1), file = file.path(outdir, "mixed_ART_alignment_check.txt"))
write_csv(as.data.frame(anova(m1)), file.path(outdir, "mixed_ART_anova.csv"))
contrasts_for(m1, "mixed")

# Model 2: fixed-effects ART
m2 <- art(coherence_median ~ crosstalk * mixture * cortical_high * band, data = d)
capture.output(summary(m2), file = file.path(outdir, "fixed_ART_alignment_check.txt"))
write_csv(as.data.frame(anova(m2)), file.path(outdir, "fixed_ART_anova.csv"))
contrasts_for(m2, "fixed")

# Descriptive statistics per cell
d %>% group_by(cortical_high, mixture, crosstalk, band) %>%
  summarise(n = n(), mean = mean(coherence_median), sd = sd(coherence_median),
            median = median(coherence_median), .groups = "drop") %>%
  write_csv(file.path(outdir, "cell_descriptives.csv"))

cat("Saved ART results to:", normalizePath(outdir), "\n")
cat("Elapsed:", format(Sys.time() - t0), "\n")
