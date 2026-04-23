library(AHSurv)
library(survival)
library(survminer)

data(ipass)

# arm: 0 = chemotherapy, 1 = gefitinib (from IPASS trial)
ipass$arm_label <- factor(ipass$arm, levels = c(0, 1),
                          labels = c("Carboplatin/Paclitaxel", "Gefitinib"))

fit <- survfit(Surv(time, status) ~ arm_label, data = ipass)

ggsurvplot(
  fit,
  data          = ipass,
  risk.table    = TRUE,
  pval          = TRUE,
  conf.int      = TRUE,
  censor = TRUE,
  xlab          = "Time (years)",
  ylab          = "Progression-Free Survival",
  title         = "PFS by Treatment Arm – IPASS Trial",
  legend.title  = "Arm",
  legend.labs   = c("Carboplatin/Paclitaxel", "Gefitinib"),
  palette       = c("#E7298A", "#1B9E77"),
  risk.table.height = 0.25,
  ggtheme       = theme_bw()
)

# ── Q1b: Hazard rate estimates ─────────────────────────────────────────────
library(muhaz)
library(bshazard)

ipass0 <- subset(ipass, arm == 0)
ipass1 <- subset(ipass, arm == 1)

mh0 <- muhaz(ipass0$time, ipass0$status)
mh1 <- muhaz(ipass1$time, ipass1$status)

suppressWarnings({
  bs0 <- bshazard(Surv(time, status) ~ 1, data = ipass0, verbose = FALSE)
  bs1 <- bshazard(Surv(time, status) ~ 1, data = ipass1, verbose = FALSE)
})

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

xlim <- range(mh0$est.grid, mh1$est.grid)
ylim <- c(0, max(mh0$haz.est, mh1$haz.est) * 1.1)
plot(mh0, col = "#E7298A", lwd = 2,
     xlim = xlim, ylim = ylim,
     xlab = "Time (months)", ylab = "Hazard rate",
     main = "Hazard Estimates (muhaz)")
lines(mh1$est.grid, mh1$haz.est, col = "#1B9E77", lwd = 2)
legend("topright", legend = c("Carboplatin/Paclitaxel", "Gefitinib"),
       col = c("#E7298A", "#1B9E77"), lwd = 2, bty = "n", cex = 0.85)

ylim2 <- c(0, max(bs0$hazard, bs1$hazard) * 1.1)
plot(bs0$time, bs0$hazard, type = "l", col = "#E7298A", lwd = 2,
     xlim = xlim, ylim = ylim2,
     xlab = "Time (months)", ylab = "Hazard rate",
     main = "Hazard Estimates (bshazard)")
lines(bs1$time, bs1$hazard, col = "#1B9E77", lwd = 2)
legend("topright", legend = c("Carboplatin/Paclitaxel", "Gefitinib"),
       col = c("#E7298A", "#1B9E77"), lwd = 2, bty = "n", cex = 0.85)

# ── Q1c: Hazard ratio over time ────────────────────────────────────────────
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# muhaz-based HR: gefitinib / chemotherapy on shared grid
t_grid <- mh0$est.grid
h0_interp <- approx(mh0$est.grid, mh0$haz.est, xout = t_grid)$y
h1_interp <- approx(mh1$est.grid, mh1$haz.est, xout = t_grid)$y
hr_muhaz  <- h1_interp / h0_interp

plot(t_grid, hr_muhaz, type = "l", lwd = 2, col = "#7570B3",
     xlab = "Time (months)", ylab = "Hazard ratio (Gefitinib / Chemo)",
     main = "Hazard Ratio over Time (muhaz)")
abline(h = 1, lty = 2, col = "gray50")

# bshazard-based HR: interpolate bs1 onto bs0 time grid
t_bs   <- bs0$time
h0_bs  <- bs0$hazard
h1_bs  <- approx(bs1$time, bs1$hazard, xout = t_bs)$y
hr_bs  <- h1_bs / h0_bs

# Remove CI calculation and filtering
plot(t_bs, hr_bs, type = "l", lwd = 2, col = "#7570B3",
     ylim = c(0, max(hr_bs, 3, na.rm = TRUE)),
     xlab = "Time (months)", ylab = "Hazard ratio (Gefitinib / Chemo)",
     main = "Hazard Ratio over Time (bshazard)")
abline(h = 1, lty = 2, col = "gray50")

# ── Q1d: Parametric distribution fits ─────────────────────────────────────
library(flexsurv)
library(RColorBrewer)

dists <- c("exp", "weibull", "lnorm", "llogis", "gamma", "gompertz")
dist_labels <- c("Exponential", "Weibull", "Log-normal",
                 "Log-logistic", "Gamma", "Gompertz")

fit_arm <- function(data) {
  lapply(dists, function(d) flexsurvreg(Surv(time, status) ~ 1, data = data, dist = d))
}

fits0 <- fit_arm(ipass0)
fits1 <- fit_arm(ipass1)

# AIC comparison table
aic_table <- function(fits, arm_name) {
  data.frame(Arm = arm_name, Dist = dist_labels, AIC = sapply(fits, AIC),
             row.names = NULL)
}
aic_df <- rbind(aic_table(fits0, "Chemo"), aic_table(fits1, "Gefitinib"))
print(aic_df[order(aic_df$Arm, aic_df$AIC), ])

# KM + parametric survival curves
km0  <- survfit(Surv(time, status) ~ 1, data = ipass0)
km1  <- survfit(Surv(time, status) ~ 1, data = ipass1)
cols <- brewer.pal(length(dists), "Dark2")

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(km0, conf.int = FALSE, col = "black", lwd = 2,
     xlab = "Time (months)", ylab = "Survival",
     main = "Parametric Fits – Carboplatin/Paclitaxel")
for (i in seq_along(fits0)) lines(fits0[[i]], col = cols[i], lwd = 1.5, ci = FALSE)
legend("topright", legend = c("KM", dist_labels),
       col = c("black", cols), lwd = c(2, rep(1.5, length(dists))),
       bty = "n", cex = 0.72)

plot(km1, conf.int = FALSE, col = "black", lwd = 2,
     xlab = "Time (months)", ylab = "Survival",
     main = "Parametric Fits – Gefitinib")
for (i in seq_along(fits1)) lines(fits1[[i]], col = cols[i], lwd = 1.5, ci = FALSE)
legend("topright", legend = c("KM", dist_labels),
       col = c("black", cols), lwd = c(2, rep(1.5, length(dists))),
       bty = "n", cex = 0.72)

# ── Q1e: Survival analysis – various methods ───────────────────────────────

## 1. Log-rank test
lr <- survdiff(Surv(time, status) ~ arm, data = ipass)
cat("\n── Log-rank test ──\n")
print(lr)

## 2. Cox proportional hazards model
cox_fit <- coxph(Surv(time, status) ~ arm, data = ipass)
cat("\n── Cox PH model ──\n")
print(summary(cox_fit))

## 3. Test PH assumption (Schoenfeld residuals)
ph_test <- cox.zph(cox_fit)
cat("\n── PH assumption test (cox.zph) ──\n")
print(ph_test)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(ph_test, main = "Schoenfeld Residuals – arm",
     xlab = "Time (months)", ylab = "Beta(t)")
abline(h = coef(cox_fit), lty = 2, col = "red")

# Log-log plot: parallel lines → PH holds
plot(survfit(Surv(time, status) ~ arm, data = ipass),
     fun = "cloglog", col = c("#E7298A", "#1B9E77"), lwd = 2,
     xlab = "log(Time)", ylab = "log(-log(S(t)))",
     main = "Log-log plot (PH check)")
legend("topleft", legend = c("Chemo", "Gefitinib"),
       col = c("#E7298A", "#1B9E77"), lwd = 2, bty = "n")

## 4. AFT models (arm as covariate) – appropriate given non-proportional hazards
aft_dists <- c("weibull", "lnorm", "llogis")
aft_fits  <- lapply(aft_dists, function(d)
  flexsurvreg(Surv(time, status) ~ arm, data = ipass, dist = d))
names(aft_fits) <- aft_dists

cat("\n── AFT models (arm covariate) – AIC ──\n")
aft_aic <- sapply(aft_fits, AIC)
print(sort(aft_aic))

cat("\n── Best AFT model summary (", names(which.min(aft_aic)), ") ──\n")
print(aft_fits[[which.min(aft_aic)]])

## 5. Royston-Parmar flexible parametric model (k=3 knots)
rp_fit <- flexsurvspline(Surv(time, status) ~ arm, data = ipass, k = 3, scale = "hazard")
cat("\n── Royston-Parmar flexible parametric model ──\n")
print(rp_fit)

par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))
plot(rp_fit, col = c("#E7298A", "#1B9E77"), lwd = 2, ci = TRUE,
     xlab = "Time (months)", ylab = "Survival",
     main = "Royston-Parmar Flexible Parametric Model")
legend("topright", legend = c("Chemo", "Gefitinib"),
       col = c("#E7298A", "#1B9E77"), lwd = 2, bty = "n")