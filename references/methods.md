# 模型速查（Quick Reference）

依據：Agresti (2019), *An Introduction to Categorical Data Analysis* (3rd ed.)。
所有數字一律用 `scripts/run_model.py` 計算。「課本 R 函式」一欄是 Agresti 書中示範的寫法，供使用者對照。

---

## 1. 二元 logistic 迴歸（§4.1–4.6）

| 項目 | 內容 |
|---|---|
| 結果變數 | 兩類（成功 = 1、失敗 = 0） |
| 模型 | $\text{logit}[\pi(x)] = \log\dfrac{\pi(x)}{1-\pi(x)} = \alpha + \beta_1x_1 + \cdots + \beta_px_p$ |
| 虛無假設 | $H_0: \beta_j = 0$（$x_j$ 對結果沒有影響，其他變數固定） |
| 統計量 | 概似比 $LR = -2(\ell_0 - \ell_1)$，$H_0$ 下 $\sim \chi^2_{df}$，$df$ = 拿掉的參數個數；Wald $z = \hat\beta/SE$ |
| 效果量 | 勝算比 $e^{\beta_j}$；預測機率 $\hat\pi(x)$ |
| 信賴區間 | profile likelihood（分離時改用 Wald），取指數得勝算比區間 |
| 預測能力 | ROC 曲線下面積 AUC（concordance index，§4.6.2） |
| Python | `statsmodels.GLM(y, X, family=Binomial())`；LR 檢定 = 兩個模型的 deviance 差 |
| 課本 R 函式 | `glm(y ~ x, family = binomial)`、`confint()`、`drop1(fit, test = "LRT")` |
| 指令 | `run_model.py logistic --csv … --y … --x … --success … --numeric …` |

解釋範本：「其他變數固定時，X 每增加 1，〈成功〉的勝算乘以 e^β（95% CI …）。」

---

## 2. Baseline-category logit（§6.1）

| 項目 | 內容 |
|---|---|
| 結果變數 | $J \ge 3$ 類，**沒有順序** |
| 模型 | $\log\dfrac{\pi_j(x)}{\pi_J(x)} = \alpha_j + \beta_j x$，$j = 1, \dots, J-1$（$J$ = 基準類別） |
| 虛無假設 | $H_0: \beta_1 = \cdots = \beta_{J-1} = 0$（x 對結果的分布沒有影響） |
| 統計量 | $LR \sim \chi^2_{J-1}$（每個解釋變數的參數個數 × 1）；個別方程式用 Wald $z$ |
| 效果量 | 每條方程式的勝算比 $e^{\beta_j}$：「第 j 類 vs 基準類別」；各類別的預測機率 |
| 注意 | 換基準類別不影響配適與預測機率；任兩類的比較 = 兩條方程式相減 |
| Python | `statsmodels.MNLogit(y, X)`（以代碼 0 的類別為基準，程式會把基準類別排在第一個） |
| 課本 R 函式 | `VGAM::vglm(y ~ x, family = multinomial(refLevel = "基準"))` |
| 指令 | `run_model.py multinomial --csv … --y … --x … --baseline …` |

---

## 3. 累積 logit／比例勝算模型（§6.2–6.3）

| 項目 | 內容 |
|---|---|
| 結果變數 | $J \ge 3$ 類，**有順序** $1 < 2 < \cdots < J$ |
| 模型 | $\text{logit}\,P(Y \le j \mid x) = \alpha_j + \beta x$，$j = 1, \dots, J-1$，所有切點**共用** $\beta$ |
| 虛無假設 | $H_0: \beta = 0$ |
| 統計量 | $LR \sim \chi^2_1$（每個解釋變數只有 1 個參數） |
| 效果量 | 「落在較高類別」的勝算比；各類別的預測機率 |
| 假設檢查 | 比例勝算：和各切點斜率不同的模型做 LR 比較，$df = (J-2) \times p$（§6.3.2） |
| 參數化 | 課本寫 $\alpha_j + \beta x$（較高類別的勝算比 $= e^{-\beta}$）；statsmodels 與 `MASS::polr` 寫 $\theta_j - \beta x$（較高類別的勝算比 $= e^{\beta}$）。`run_model.py` 輸出的 `or_higher` 一律是「落在較高類別」的方向 |
| 注意 | 不要把有序結果壓成兩類 |
| Python | `statsmodels.miscmodels.ordinal_model.OrderedModel(y, X, distr="logit")`；比例勝算檢定的一般模型由 `run_model.py` 以最大概似法自行配適 |
| 課本 R 函式 | `VGAM::vglm(y ~ x, family = cumulative(parallel = TRUE))` |
| 指令 | `run_model.py ordinal --csv … --y … --x … --levels 低,…,高` |

---

## 4. 比較兩個模型（§4.3、§5.1–5.2）

| 項目 | 內容 |
|---|---|
| 用途 | 數字代碼當數值還是類別（§4.3）；要不要放某個變數；兩組不同的解釋變數哪個好 |
| 巢狀模型 | 一個模型是另一個的特例（例如數值分數模型 ⊂ 類別模型；少一個變數的模型 ⊂ 完整模型） |
| 虛無假設 | $H_0$：較簡單的模型就足夠（多出來的參數都是 0 或符合限制） |
| 統計量 | $LR = -2(\ell_{\text{簡單}} - \ell_{\text{複雜}})$，$H_0$ 下 $\sim \chi^2_{df}$，$df$ = 參數個數差 |
| 不是巢狀 | 不能做 LR 檢定，只比較 $AIC = -2\ell + 2k$，越小越好（§5.1） |
| 注意 | 兩個模型必須使用同一批觀察值（程式會依兩個模型的所有變數一起排除缺失值） |
| 課本 R 函式 | `anova(fit0, fit1, test = "LRT")`、`AIC(fit)` |
| 指令 | `run_model.py compare --type logistic --csv … --y … --x … --numeric … --x2 … --numeric2 …` |

解釋範本：「把 failures 改當類別後，LR = …，df = …，P = …：類別模型明顯較好，代表『每多被當一次效果相同』的假設不成立。」

---

## 決策規則（所有模型共用）

| 檢定 | 拒絕 $H_0$ 若 |
|---|---|
| 概似比檢定 | $LR \ge \chi^2_{df,\alpha}$（例如 $df = 1$、$\alpha = 0.05$ 時為 3.84） |
| Wald 檢定（雙尾） | $\lvert z \rvert \ge z_{\alpha/2}$ |
| 勝算比信賴區間 | 區間不包含 1 ⇔ 雙尾檢定拒絕 $H_0$ |
