# 新程式（只用 numpy、pandas）和 statsmodels／R 的比對結果

由 `tests/test_against_reference.py` 產生。Skill 一律在**擋掉 statsmodels、scipy、scikit-learn** 的情況下執行。
差距 = |新 − 參考| / max(1, |參考|)，每一項都要小於 1e-6。

**結果：146 / 146 項通過**（共比對 1571 個數字）

| 例子 | 項目 | 數字個數 | 最大差距 | 參考 |
|---|---|---|---|---|
| 馬蹄蟹 y ~ width（§4.1） | 係數 β | 2 | 7.2e-16 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | 標準誤 | 2 | 3.0e-11 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | Wald z | 2 | 3.1e-11 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | Wald P | 2 | 1.8e-15 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | 勝算比 | 2 | 4.1e-16 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | profile CI 下界（log 勝算比） | 2 | 2.8e-15 | statsmodels + scipy brentq |
| 馬蹄蟹 y ~ width（§4.1） | profile CI 上界（log 勝算比） | 2 | 3.6e-15 | statsmodels + scipy brentq |
| 馬蹄蟹 y ~ width（§4.1） | 係數 β | 2 | 1.6e-15 | R glm |
| 馬蹄蟹 y ~ width（§4.1） | 標準誤 | 2 | 3.0e-11 | R glm |
| 馬蹄蟹 y ~ width（§4.1） | 各變數 LR | 1 | 1.8e-15 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | 各變數 LR 的 P | 1 | 6.3e-22 | statsmodels + scipy |
| 馬蹄蟹 y ~ width（§4.1） | 各變數 LR | 1 | 5.7e-16 | R glm |
| 馬蹄蟹 y ~ width（§4.1） | 整體 LR | 1 | 1.8e-15 | statsmodels |
| 馬蹄蟹 y ~ width（§4.1） | 整體 LR | 1 | 5.7e-16 | R glm |
| 馬蹄蟹 y ~ width（§4.1） | 整體 LR 的 P | 1 | 6.3e-22 | scipy |
| 馬蹄蟹 y ~ width（§4.1） | AUC | 1 | 0.0e+00 | scikit-learn |
| 馬蹄蟹 y ~ width（§4.1） | AUC | 1 | 0.0e+00 | R pROC |
| 馬蹄蟹 y ~ width（§4.1） | 預測機率 | 2 | 3.3e-16 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | 係數 β | 5 | 4.6e-15 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | 標準誤 | 5 | 2.2e-14 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | Wald z | 5 | 2.2e-14 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | Wald P | 5 | 4.8e-15 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | 勝算比 | 5 | 3.6e-15 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | profile CI 下界（log 勝算比） | 5 | 6.6e-15 | statsmodels + scipy brentq |
| 馬蹄蟹 y ~ width + color（§4.4） | profile CI 上界（log 勝算比） | 5 | 8.2e-14 | statsmodels + scipy brentq |
| 馬蹄蟹 y ~ width + color（§4.4） | 係數 β | 5 | 5.1e-15 | R glm |
| 馬蹄蟹 y ~ width + color（§4.4） | 標準誤 | 5 | 3.4e-10 | R glm |
| 馬蹄蟹 y ~ width + color（§4.4） | 各變數 LR | 2 | 2.3e-15 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | 各變數 LR 的 P | 2 | 1.7e-16 | statsmodels + scipy |
| 馬蹄蟹 y ~ width + color（§4.4） | 各變數 LR | 2 | 7.2e-16 | R glm |
| 馬蹄蟹 y ~ width + color（§4.4） | 整體 LR | 1 | 1.5e-15 | statsmodels |
| 馬蹄蟹 y ~ width + color（§4.4） | 整體 LR | 1 | 3.7e-16 | R glm |
| 馬蹄蟹 y ~ width + color（§4.4） | 整體 LR 的 P | 1 | 2.6e-21 | scipy |
| 馬蹄蟹 y ~ width + color（§4.4） | AUC | 1 | 2.2e-16 | scikit-learn |
| 馬蹄蟹 y ~ width + color（§4.4） | AUC | 1 | 4.4e-16 | R pROC |
| 馬蹄蟹 y ~ width + color（§4.4） | 預測機率 | 6 | 1.2e-15 | statsmodels |
| 鱷魚 food ~ length（§6.1） | 係數 β | 4 | 6.9e-16 | statsmodels |
| 鱷魚 food ~ length（§6.1） | 標準誤 | 4 | 9.3e-15 | statsmodels |
| 鱷魚 food ~ length（§6.1） | Wald z | 4 | 9.6e-15 | statsmodels |
| 鱷魚 food ~ length（§6.1） | Wald P | 4 | 1.8e-15 | scipy |
| 鱷魚 food ~ length（§6.1） | Wald CI 下界（log 勝算比） | 4 | 1.6e-14 | statsmodels |
| 鱷魚 food ~ length（§6.1） | Wald CI 上界（log 勝算比） | 4 | 1.6e-14 | statsmodels |
| 鱷魚 food ~ length（§6.1） | 係數 β | 4 | 5.8e-09 | R VGAM |
| 鱷魚 food ~ length（§6.1） | 標準誤 | 4 | 3.5e-09 | R VGAM |
| 鱷魚 food ~ length（§6.1） | 整體 LR | 1 | 1.7e-15 | statsmodels |
| 鱷魚 food ~ length（§6.1） | 整體 LR | 1 | 2.1e-15 | R VGAM |
| 鱷魚 food ~ length（§6.1） | 各變數 LR | 1 | 1.7e-15 | statsmodels |
| 鱷魚 food ~ length（§6.1） | 各變數 LR | 1 | 2.1e-15 | R VGAM |
| 鱷魚 food ~ length（§6.1） | 各變數 LR 的 P | 1 | 3.1e-18 | scipy |
| 鱷魚 food ~ length（§6.1） | 預測機率 | 6 | 1.1e-16 | statsmodels |
| 政治意識形態 ~ party + gender（§6.2） | 係數 β | 2 | 8.3e-15 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | 標準誤 | 2 | 1.8e-15 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | Wald z | 2 | 5.4e-14 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | Wald P | 2 | 4.1e-14 | R + scipy |
| 政治意識形態 ~ party + gender（§6.2） | 較高類別的勝算比 | 2 | 8.0e-15 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | Wald CI 下界（log 勝算比） | 2 | 9.4e-15 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | Wald CI 上界（log 勝算比） | 2 | 7.2e-15 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | 整體 LR | 1 | 8.4e-16 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | 各變數 LR | 2 | 2.3e-13 | R ordinal::clm |
| 政治意識形態 ~ party + gender（§6.2） | 各變數 LR 的 P | 2 | 2.7e-13 | scipy |
| 政治意識形態 ~ party + gender（§6.2） | 比例勝算檢定 LR | 1 | 9.0e-12 | R ordinal::clm（nominal） |
| 政治意識形態 ~ party + gender（§6.2） | 比例勝算檢定 P | 1 | 6.0e-12 | scipy |
| 政治意識形態 ~ party + gender（§6.2） | 預測機率 | 20 | 2.0e-15 | R ordinal::clm |
| 高中生 pass（實測第 1 題） | 係數 β | 5 | 9.4e-16 | statsmodels |
| 高中生 pass（實測第 1 題） | 標準誤 | 5 | 3.2e-13 | statsmodels |
| 高中生 pass（實測第 1 題） | Wald z | 5 | 1.9e-12 | statsmodels |
| 高中生 pass（實測第 1 題） | Wald P | 5 | 3.5e-13 | statsmodels |
| 高中生 pass（實測第 1 題） | 勝算比 | 5 | 9.2e-16 | statsmodels |
| 高中生 pass（實測第 1 題） | profile CI 下界（log 勝算比） | 5 | 1.3e-13 | statsmodels + scipy brentq |
| 高中生 pass（實測第 1 題） | profile CI 上界（log 勝算比） | 5 | 2.2e-13 | statsmodels + scipy brentq |
| 高中生 pass（實測第 1 題） | 係數 β | 5 | 6.2e-15 | R glm |
| 高中生 pass（實測第 1 題） | 標準誤 | 5 | 3.2e-13 | R glm |
| 高中生 pass（實測第 1 題） | 各變數 LR | 4 | 5.7e-14 | statsmodels |
| 高中生 pass（實測第 1 題） | 各變數 LR 的 P | 4 | 2.8e-14 | statsmodels + scipy |
| 高中生 pass（實測第 1 題） | 各變數 LR | 4 | 1.1e-14 | R glm |
| 高中生 pass（實測第 1 題） | 整體 LR | 1 | 2.3e-15 | statsmodels |
| 高中生 pass（實測第 1 題） | 整體 LR | 1 | 3.1e-15 | R glm |
| 高中生 pass（實測第 1 題） | 整體 LR 的 P | 1 | 5.1e-33 | scipy |
| 高中生 pass（實測第 1 題） | AUC | 1 | 1.1e-16 | scikit-learn |
| 高中生 pass（實測第 1 題） | AUC | 1 | 0.0e+00 | R pROC |
| 高中生 pass（實測第 1 題） | 預測機率 | 8 | 2.8e-16 | statsmodels |
| 高中生 pass，failures 當類別 | 係數 β | 7 | 4.4e-15 | statsmodels |
| 高中生 pass，failures 當類別 | 標準誤 | 7 | 6.7e-16 | statsmodels |
| 高中生 pass，failures 當類別 | Wald z | 7 | 5.8e-15 | statsmodels |
| 高中生 pass，failures 當類別 | Wald P | 7 | 2.5e-15 | statsmodels |
| 高中生 pass，failures 當類別 | 勝算比 | 7 | 5.5e-15 | statsmodels |
| 高中生 pass，failures 當類別 | profile CI 下界（log 勝算比） | 7 | 3.4e-13 | statsmodels + scipy brentq |
| 高中生 pass，failures 當類別 | profile CI 上界（log 勝算比） | 7 | 2.0e-13 | statsmodels + scipy brentq |
| 高中生 pass，failures 當類別 | 係數 β | 7 | 5.9e-15 | R glm |
| 高中生 pass，failures 當類別 | 標準誤 | 7 | 9.0e-13 | R glm |
| 高中生 pass，failures 當類別 | 各變數 LR | 4 | 5.7e-14 | statsmodels |
| 高中生 pass，failures 當類別 | 各變數 LR 的 P | 4 | 1.6e-14 | statsmodels + scipy |
| 高中生 pass，failures 當類別 | 各變數 LR | 4 | 5.7e-14 | R glm |
| 高中生 pass，failures 當類別 | 整體 LR | 1 | 2.1e-15 | statsmodels |
| 高中生 pass，failures 當類別 | 整體 LR | 1 | 5.3e-16 | R glm |
| 高中生 pass，failures 當類別 | 整體 LR 的 P | 1 | 1.3e-33 | scipy |
| 高中生 pass，failures 當類別 | AUC | 1 | 0.0e+00 | scikit-learn |
| 高中生 pass，failures 當類別 | AUC | 1 | 0.0e+00 | R pROC |
| 高中生 pass，failures 當類別 | 預測機率 | 10 | 1.0e-15 | statsmodels |
| 高中生 Walc ~ sex（實測第 2 題） | 係數 β | 1 | 3.5e-15 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | 標準誤 | 1 | 3.6e-16 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | Wald z | 1 | 5.8e-15 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | Wald P | 1 | 1.0e-26 | R + scipy |
| 高中生 Walc ~ sex（實測第 2 題） | 較高類別的勝算比 | 1 | 4.0e-15 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | Wald CI 下界（log 勝算比） | 1 | 4.7e-15 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | Wald CI 上界（log 勝算比） | 1 | 2.5e-15 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | 整體 LR | 1 | 0.0e+00 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | 各變數 LR | 1 | 0.0e+00 | R ordinal::clm |
| 高中生 Walc ~ sex（實測第 2 題） | 各變數 LR 的 P | 1 | 7.9e-30 | scipy |
| 高中生 Walc ~ sex（實測第 2 題） | 比例勝算檢定 LR | 1 | 1.5e-14 | R ordinal::clm（nominal） |
| 高中生 Walc ~ sex（實測第 2 題） | 比例勝算檢定 P | 1 | 8.0e-17 | scipy |
| 高中生 Walc ~ sex（實測第 2 題） | 預測機率 | 10 | 8.9e-16 | R ordinal::clm |
| 高中生 reason ~ address + school（實測第 3 題） | 係數 β | 9 | 2.2e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | 標準誤 | 9 | 2.7e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | Wald z | 9 | 8.3e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | Wald P | 9 | 4.8e-15 | scipy |
| 高中生 reason ~ address + school（實測第 3 題） | Wald CI 下界（log 勝算比） | 9 | 4.2e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | Wald CI 上界（log 勝算比） | 9 | 4.3e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | 係數 β | 9 | 4.9e-11 | R VGAM |
| 高中生 reason ~ address + school（實測第 3 題） | 標準誤 | 9 | 3.6e-12 | R VGAM |
| 高中生 reason ~ address + school（實測第 3 題） | 整體 LR | 1 | 0.0e+00 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | 整體 LR | 1 | 6.6e-16 | R VGAM |
| 高中生 reason ~ address + school（實測第 3 題） | 各變數 LR | 2 | 5.2e-15 | statsmodels |
| 高中生 reason ~ address + school（實測第 3 題） | 各變數 LR | 2 | 5.6e-15 | R VGAM |
| 高中生 reason ~ address + school（實測第 3 題） | 各變數 LR 的 P | 2 | 1.6e-22 | scipy |
| 高中生 reason ~ address + school（實測第 3 題） | 預測機率 | 16 | 3.1e-16 | statsmodels |
| 模型比較（logistic） | 對數概似 | 2 | 3.8e-16 | statsmodels |
| 模型比較（logistic） | deviance | 2 | 0.0e+00 | statsmodels |
| 模型比較（logistic） | AIC | 2 | 3.7e-16 | statsmodels |
| 模型比較（logistic） | AIC | 2 | 1.2e-15 | R glm |
| 模型比較（logistic） | LR 檢定與 P | 2 | 1.8e-14 | statsmodels + scipy |
| 模型比較（累積 logit） | 對數概似與 AIC | 4 | 0.0e+00 | R ordinal::clm |
| 模型比較（累積 logit） | LR 檢定 | 1 | 0.0e+00 | R ordinal::clm |
| 沒有 matplotlib | 所有係數（和有 matplotlib 時相比） | 5 | 0.0e+00 | 同一程式 |
| 沒有 matplotlib | 照常輸出 result.json、不出圖 | 1 | 0.0e+00 | — |
| 完全分離（logistic） | 警告 SEPARATION、x 的估計值輸出 Inf、SE／P／CI 輸出 NA | 1 | 0.0e+00 | §5.3 |
| 完全分離（logistic） | 整體 LR（極限值 = 虛無 deviance = 12 ln 2） | 1 | 2.1e-16 | 精確值 |
| 準完全分離（logistic） | 警告 SEPARATION、x 的估計值輸出 Inf | 1 | 0.0e+00 | §5.3 |
| 準完全分離（logistic） | 整體 LR（極限值 = 16 ln 2 − 4 ln 2） | 1 | 1.2e-14 | 精確值 |
| 分離（baseline-category logit） | 只有 gb:1（A vs C）輸出 −Inf，其他係數正常 | 1 | 0.0e+00 | §5.3 |
| 分離（累積 logit） | 警告 SEPARATION、gb 的估計值輸出 Inf | 1 | 0.0e+00 | §5.3 |
| 分布函數 | 卡方右尾機率（df 1–30，420 個點） | 420 | 4.3e-15 | scipy |
| 分布函數 | 常態右尾機率（−8 到 8） | 161 | 1.1e-16 | scipy |
| 分布函數 | 卡方分位數（0.95，df 1–10） | 10 | 3.4e-16 | scipy |
| 分布函數 | 等級（含同分，AUC 用） | 500 | 0.0e+00 | scipy |
| check 指令 | 擋掉 statsmodels、scipy 時「全部通過」 | 1 | 0.0e+00 | — |

## 說明

- 政治意識形態 ~ party + gender（§6.2）：statsmodels OrderedModel 和 R clm 的差距：β 7.7e-09，標準誤 2.1e-07（OrderedModel 用數值微分，所以拿 clm 當主要參考值）
- 高中生 Walc ~ sex（實測第 2 題）：statsmodels OrderedModel 和 R clm 的差距：β 1.5e-09，標準誤 3.1e-08（OrderedModel 用數值微分，所以拿 clm 當主要參考值）
- 卡方右尾機率的最大「相對」誤差 2.7e-14（連 1e-100 這麼小的 P 值也一樣準）

## 和舊版（statsmodels）實測結果的比較

舊版實測時存下的 `result.json`，和新版重跑的結果比較。舊版的收斂標準較寬（statsmodels GLM 預設 1e-8、OrderedModel 用 BFGS 與數值微分），
所以差距在 1e-6 到 1e-4 之間；新版和收斂到 1e-14 的 statsmodels、R 的差距都小於 1e-6（上表），新版比舊版更準。

| 舊版結果 | 數字個數 | 最大差距 | 差距最大的項目 |
|---|---|---|---|
| `round1/q1` | 54 | 1.2e-06 | `/coefficients[1]/z` |
| `round1/q2` | 23 | 4.8e-05 | `/coefficients[0]/or_upper` |
| `round1/q3` | 85 | 1.0e-14 | `/coefficients[7]/z` |
| `round2/q1_cat` | 70 | 1.9e-06 | `/coefficients[1]/z` |
| `round2/q1_compare` | 8 | 6.0e-15 | `/lr_test/lr` |
