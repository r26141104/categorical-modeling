# Part 2：獨立驗證計算
#
# 用來核對 Skill 在真實資料上的數字。刻意「不使用」本專案 R/ 資料夾的任何函式，
# 並盡量用和 Skill 不同的套件計算（Skill 用 VGAM；這裡用 MASS、ordinal、nnet）。
# 設定（例如 studytime 當數值或類別、基準類別）要和 Skill 實測時使用者的回答一致；
# 實測後若設定不同，請依 spec.json 調整這裡再重跑。
#
# 執行（在專案根目錄）：Rscript demo/part2/verify.R
# 輸出：demo/part2/verify_results.json

d <- read.csv("data/uci-student/student_por.csv")
out <- list()

# ---- q1：二元 logistic（studytime 當數值分數；higher、sex 為類別）----
d$pass01 <- as.integer(d$pass == "pass")
d$higher <- factor(d$higher, levels = c("no", "yes"))
d$sex <- factor(d$sex, levels = c("F", "M"))
fit1 <- glm(pass01 ~ studytime + failures + higher + sex, family = binomial, data = d)
lr1 <- car::Anova(fit1, type = "II", test.statistic = "LR")
out$q1 <- list(
  coef = as.list(coef(fit1)),
  odds_ratio = as.list(exp(coef(fit1))),
  LR = as.list(setNames(lr1[, "LR Chisq"], rownames(lr1))),
  LR_p = as.list(setNames(lr1[, "Pr(>Chisq)"], rownames(lr1))),
  auc = as.numeric(pROC::auc(pROC::roc(d$pass01, fitted(fit1), quiet = TRUE, direction = "<"))))

# ---- q1 追問：failures 當數值分數 vs 當類別（巢狀模型的概似比檢定，§4.3、§5.2）----
# 數值分數只用 1 個參數、假設每多被當一次效果相同；當類別時 0、1、2、3 各自估計（3 個參數）
fit1_cat <- glm(pass01 ~ studytime + factor(failures) + higher + sex, family = binomial, data = d)
cmp1 <- anova(fit1, fit1_cat, test = "LRT")
out$q1_compare <- list(
  loglik_numeric = as.numeric(logLik(fit1)), loglik_categorical = as.numeric(logLik(fit1_cat)),
  AIC_numeric = AIC(fit1), AIC_categorical = AIC(fit1_cat),
  LR = cmp1$Deviance[2], df = cmp1$Df[2], p = cmp1[["Pr(>Chi)"]][2])

# ---- q2：有序 logistic（累積 logit、比例勝算）----
# MASS::polr 的參數化是 logit P(Y <= j) = zeta_j - beta * x，beta > 0 表示傾向較高類別
d$Walc_o <- factor(d$Walc, levels = 1:5, ordered = TRUE)
fit2 <- MASS::polr(Walc_o ~ sex, data = d, Hess = TRUE)
fit2_0 <- MASS::polr(Walc_o ~ 1, data = d, Hess = TRUE)
clm2 <- ordinal::clm(Walc_o ~ sex, data = d)
po2 <- ordinal::nominal_test(clm2)   # 比例勝算假設的 LR 檢定
out$q2 <- list(
  beta_sexM = unname(coef(fit2)["sexM"]),
  odds_ratio_higher_sexM = unname(exp(coef(fit2)["sexM"])),
  LR = fit2_0$deviance - fit2$deviance,
  PO_LR = po2[["LRT"]][2], PO_df = po2[["Df"]][2], PO_p = po2[["Pr(>Chi)"]][2])

# ---- q3：多類別（baseline-category）logistic，基準類別 = other ----
d$reason <- relevel(factor(d$reason), ref = "other")
d$address <- factor(d$address, levels = c("R", "U"))
d$school <- factor(d$school, levels = c("GP", "MS"))
fit3 <- nnet::multinom(reason ~ address + school, data = d, trace = FALSE)
fit3_0 <- nnet::multinom(reason ~ 1, data = d, trace = FALSE)
fit3_a <- nnet::multinom(reason ~ school, data = d, trace = FALSE)
fit3_s <- nnet::multinom(reason ~ address, data = d, trace = FALSE)
cf3 <- coef(fit3)
out$q3 <- list(
  coef = lapply(split(cf3, rownames(cf3)), function(r) as.list(setNames(r, colnames(cf3)))),
  LR_all = deviance(fit3_0) - deviance(fit3),
  LR_address = deviance(fit3_a) - deviance(fit3),
  LR_school = deviance(fit3_s) - deviance(fit3))

jsonlite::write_json(out, "demo/part2/verify_results.json", auto_unbox = TRUE, digits = NA, pretty = TRUE)
str(out, give.attr = FALSE)
