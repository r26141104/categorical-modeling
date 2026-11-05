# 產生 R 的參考值（tests/reference_R.json），給 test_against_reference.py 比對用。
# 只在本機測試時使用；Skill 執行時不需要 R。
# 收斂標準都設到 1e-15：R 預設的收斂標準較寬（glm 的 epsilon = 1e-8），標準誤只準到約 1e-5。
#
# 執行（在 repo 根目錄）：Rscript tests/make_reference_R.R
# 需要套件：VGAM、ordinal、pROC、jsonlite

suppressMessages({ library(VGAM); library(ordinal); library(pROC); library(jsonlite) })
gctl <- glm.control(epsilon = 1e-15, maxit = 200)
vctl <- vglm.control(epsilon = 1e-15, maxit = 200)
cctl <- clm.control(gradTol = 1e-12, maxIter = 500)
out <- list()

coefs <- function(est, se) list(names = names(est), estimate = unname(est), se = unname(se))

glm_case <- function(fml, data, drops) {
  f <- glm(fml, family = binomial, data = data, control = gctl)
  f0 <- glm(update(fml, . ~ 1), family = binomial, data = data, control = gctl)
  lr <- sapply(drops, function(v) {
    fr <- glm(update(fml, as.formula(paste(". ~ . -", v))), family = binomial, data = data, control = gctl)
    deviance(fr) - deviance(f)
  })
  c(coefs(coef(f), sqrt(diag(vcov(f)))),
    list(deviance = deviance(f), loglik = as.numeric(logLik(f)), overall_lr = deviance(f0) - deviance(f),
         lr = as.list(lr), auc = as.numeric(auc(roc(f$y, round(fitted(f), 12), quiet = TRUE, direction = "<")))))  # 四捨五入：解釋變數相同的人同分
}

# ---- 二元 logistic：課本馬蹄蟹（§4.1、§4.4）與示範資料（高中生）
crabs <- read.csv("data/crabs.csv")
out$crabs_width <- glm_case(y ~ width, crabs, c("width"))
crabs$color <- factor(crabs$color)
out$crabs_color <- glm_case(y ~ width + color, crabs, c("width", "color"))
st <- read.csv("data/student_por.csv")
st$passy <- as.integer(st$pass == "pass")
st$higher <- factor(st$higher, levels = c("no", "yes")); st$sex <- factor(st$sex, levels = c("F", "M"))
out$student_q1 <- glm_case(passy ~ studytime + failures + higher + sex, st, c("studytime", "failures", "higher", "sex"))
st$failuresf <- factor(st$failures)
out$student_q1cat <- glm_case(passy ~ studytime + failuresf + higher + sex, st, c("studytime", "failuresf", "higher", "sex"))

# ---- baseline-category logit：課本鱷魚（§6.1）與示範資料（選校原因）
mn_case <- function(fml, data, drops) {
  f <- vglm(fml, family = multinomial(refLevel = 1), data = data, control = vctl)
  f0 <- vglm(update(fml, . ~ 1), family = multinomial(refLevel = 1), data = data, control = vctl)
  lr <- sapply(drops, function(v) {
    fr <- vglm(update(fml, as.formula(paste(". ~ . -", v))), family = multinomial(refLevel = 1), data = data, control = vctl)
    2 * (logLik(f) - logLik(fr))
  })
  c(coefs(coef(f), sqrt(diag(vcov(f)))),
    list(loglik = as.numeric(logLik(f)), overall_lr = 2 * as.numeric(logLik(f) - logLik(f0)), lr = as.list(lr)))
}
al <- read.csv("data/alligators.csv")
al$food <- factor(al$food, levels = c("O", "F", "I"))           # 基準類別 O 排第一
out$alligators <- mn_case(food ~ length, al, c("length"))
st$reason <- factor(st$reason, levels = c("other", "course", "home", "reputation"))
st$address <- factor(st$address); st$school <- factor(st$school)
out$student_q3 <- mn_case(reason ~ address + school, st, c("address", "school"))

# ---- 累積 logit：課本政治意識形態（§6.2，交叉表展開）與示範資料（週末飲酒量）
clm_case <- function(fml, data, drops, nominal) {
  f <- clm(fml, data = data, link = "logit", control = cctl)
  y <- model.response(model.frame(fml, data))
  ll0 <- sum(table(y) * log(table(y) / length(y)))
  lr <- sapply(drops, function(v) {
    fr <- clm(update(fml, as.formula(paste(". ~ . -", v))), data = data, link = "logit", control = cctl)
    2 * (logLik(f) - logLik(fr))
  })
  fn <- clm(fml, nominal = nominal, data = data, link = "logit", control = cctl)   # 非比例勝算（每個切點各有斜率）
  b <- coef(f)[!grepl("\\|", names(coef(f)))]
  th <- coef(f)[grepl("\\|", names(coef(f)))]
  se <- sqrt(diag(vcov(f)))
  c(coefs(b, se[names(b)]),
    list(theta = unname(th), se_theta = unname(se[names(th)]), loglik = as.numeric(logLik(f)),
         overall_lr = 2 * (as.numeric(logLik(f)) - ll0), lr = as.list(lr), po_lr = 2 * as.numeric(logLik(fn) - logLik(f))))
}
pol <- read.csv("report/skill-test/tidy-check/polviews_tidy.csv", fileEncoding = "UTF-8-BOM")
pol$ideology <- factor(pol$ideology, levels = paste0("y", 1:5), ordered = TRUE)
pol$party <- factor(pol$party); pol$gender <- factor(pol$gender)
out$polviews <- clm_case(ideology ~ party + gender, pol, c("party", "gender"), ~ party + gender)
st$Walc <- factor(st$Walc, levels = 1:5, ordered = TRUE)
out$student_q2 <- clm_case(Walc ~ sex, st, c("sex"), ~ sex)
st$address <- factor(st$address)
out$student_q2_addr <- list(loglik = as.numeric(logLik(clm(Walc ~ sex + address, data = st, link = "logit", control = cctl))))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "tests/reference_R.json")
cat("已寫出 tests/reference_R.json\n")
