# 對話紀錄：影響學生及格的因素（二元 logistic 迴歸）

- 日期：2026-10-28
- 資料：student_por.csv（649 列 × 34 欄）
- 使用工具：categorical-modeling skill（`scripts/run_model.py`，Python statsmodels）
- 學術依據：Agresti, *An Introduction to Categorical Data Analysis* (3rd ed., 2019)

---

## 1. 使用者

> /categorical-modeling
> 用 categorical-modeling skill 分析這份資料：讀書時間（studytime）、被當次數（failures）、想不想讀大學（higher）、性別（sex），哪些會影響是否及格（pass）？

## 1. Claude：資料描述與確認清單

執行指令：

```bash
python run_model.py describe --csv student_por.csv --vars pass,studytime,failures,higher,sex
```

### 資料概況

共 649 位學生，這 5 個變數都沒有缺失值。

| 變數 | 次數 |
|---|---|
| pass | fail 100、pass 549 |
| studytime | 1：212、2：305、3：97、4：35 |
| failures | 0：549、1：70、2：16、3：14 |
| higher | no 69、yes 580 |
| sex | F 383、M 266 |

### 確認清單（括號內為建議的預設）

1. **資料結構**：每一列是不同、彼此獨立的學生（需由使用者確認，§1.1.3）。
2. **結果變數**：兩類，fail / pass。
3. **數字代碼變數的用法**（§4.3）：studytime、failures 都建議當數值分數。
4. **參考組**：higher 以 no 為參考組；sex 以 F 為參考組。
5. **成功類別**：pass。
6. **樣本數檢查**：較少的一類有 100 人，預設模型有 5 個參數，平均每個參數 20 人，足夠（§5.1.1）。

**推薦的模型**：二元 logistic 迴歸（§4.2–4.4）。用概似比檢定逐一檢定各變數，H₀ 為 β = 0。

---

## 2. 使用者

> 每位學生彼此獨立。studytime 和 failures 當數值分數，其他照預設。

## 2. Claude：配適模型 A

執行指令：

```bash
python run_model.py logistic --csv student_por.csv --y pass --x studytime,failures,higher,sex \
  --success pass --numeric studytime,failures --ref higher=no --ref sex=F --out output/q1
```

### 整體檢定

四個變數合起來和「只有截距」的模型比較：LR = 96.82，df = 4，P < 0.001。

### 係數與勝算比

| 變數 | β（SE） | 勝算比 | 95% CI | LR 檢定 P |
|---|---|---|---|---|
| studytime（每增加 1） | 0.372（0.170） | 1.450 | 1.050 – 2.046 | 0.024 |
| failures（每增加 1） | -1.028（0.171） | 0.358 | 0.253 – 0.495 | < 0.001 |
| higher = yes（相對於 no） | 1.326（0.309） | 3.766 | 2.040 – 6.877 | < 0.001 |
| sex = M（相對於 F） | -0.164（0.249） | 0.848 | 0.521 – 1.388 | 0.510 |

### 預測能力

AUC = 0.799。

### 預測機率

其他變數固定在平均數或最常見類別時的及格機率：

| 情況 | 及格機率 |
|---|---|
| studytime = 1 | 0.861 |
| studytime = 2 | 0.900 |
| failures = 0 | 0.917 |
| failures = 3 | 0.335 |
| higher = no | 0.699 |
| higher = yes | 0.898 |
| sex = F | 0.898 |
| sex = M | 0.881 |

### 解釋

- **結論**：被當次數、想不想讀大學、讀書時間都和是否及格有關；性別則沒有。
- **failures 影響最大**：每多被當一次，及格勝算約乘以 0.36。
- **higher**：想讀大學的學生，及格勝算約是不想讀的 3.8 倍。
- **studytime**：每多一級，及格勝算乘以 1.45。信賴區間下限接近 1，效果比較不確定。

### 注意事項

- **[GOF_NOTE]**：模型含數值變數，deviance 不能當適合度檢定（§5.2.3）。
- **等距假設**：failures 當數值，等於假設每多被當一次的效果相同，這個假設不一定成立。
- **資料本質**：這是觀察資料，結果是關聯，不是因果。
- **常見誤解**：勝算比不是機率的倍數。

---

## 3. 使用者

> 好，比較 failures 當數值和當類別。

## 3. Claude：模型比較

執行指令：

```bash
python run_model.py compare --type logistic --csv student_por.csv --y pass \
  --x studytime,failures,higher,sex --numeric studytime,failures --numeric2 studytime \
  --success pass --ref higher=no --ref sex=F --out output/q1_compare
```

### 比較結果

| 模型 | 參數 | deviance | AIC |
|---|---|---|---|
| A：failures 數值 | 5 | 460.97 | 470.97 |
| B：failures 類別 | 7 | 451.44 | 465.44 |

- **概似比檢定**：LR = 9.53，df = 2，P = 0.009。
- **AIC**：模型 B 較小。

### 結論

模型 B 明顯比較好，表示 failures 的等距假設不成立（§4.3、§5.2）。

改當類別後要注意：被當 2 次（16 人）和 3 次（14 人）這兩組人數很少，估計可能不穩定（§5.1.1）。

---

## 4. 使用者

> 產生報告。

## 4. Claude：產生報告

1. 撰寫 `summary.md`（三段摘要，只使用程式輸出的數字）。
2. 執行以下指令，產出 `output/report.html`：

```bash
python run_model.py report --out output/q1,output/q1_compare \
  --title "影響學生及格的因素：二元 logistic 迴歸分析" --summary summary.md --file output/report.html
```

提醒：報告中的勝算比來自模型 A。如需 failures 各類別的效果，可以用模型 B 重新配適（尚未執行）。

---

## 5. 使用者

> 請把所有 result.json、圖檔和 report.html 打包成一個 zip，並把這段對話整理成 transcript.md 給我下載。

## 5. Claude

打包成 `analysis_results.zip`，並提供本檔 `transcript.md`。

---

> 以下第 6、7 輪是 2026-10-29 在同一個對話中追加的；依當時的回覆截圖整理（`screenshots/refit-*.png`），數字以 `q1_cat/result.json` 為準。

## 6. 使用者

> 用 failures 當類別重新配適第 1 題，其他設定不變（成功類別 pass，studytime 當數值，higher 參考組 no，sex 參考組 F）。

## 6. Claude：重新配適（模型 B，輸出 `output/q1_cat`）

模型 B 已重新配適，設定是 failures 當類別、以 0 次為參考組，其他不變。被當次數的效果主要在「有沒有被當過」：從 0 次到 1 次落差最大，之後的差異就小很多。

### 一句話結論

結論和模型 A 相同：被當次數、想不想讀大學、讀書時間都和是否及格有關，性別沒有。依據是概似比檢定（§4.4）：

| 變數 | LR | df | P |
|---|---|---|---|
| failures | 49.04 | 3 | < 0.001 |
| higher | 14.95 | 1 | < 0.001 |
| studytime | 4.11 | 1 | 0.043 |
| sex | 0.85 | 1 | 0.358 |

四個變數合起來和只有截距的模型比較：LR = 106.35，df = 6，P < 0.001。

### failures 各類別的效果（以被當 0 次為基準，固定其他變數）

| 被當次數 | 人數 | 勝算比 | 95% CI | 預測及格機率 |
|---|---|---|---|---|
| 0 | 549 | 1（參考組） | — | 0.927 |
| 1 | 70 | 0.164 | 0.091 – 0.295 | 0.674 |
| 2 | 16 | 0.158 | 0.052 – 0.484 | 0.666 |
| 3 | 14 | 0.099 | 0.028 – 0.319 | 0.555 |

- 0 → 1 次的落差最大：及格勝算剩約六分之一，預測機率從 0.927 掉到 0.674。
- 1 次和 2 次幾乎一樣；3 次再低一些。
- 這解釋了模型 A 不適合的原因：模型 A 假設每多被當一次，勝算都乘以同一個數（0.358），所以會低估「第一次被當」的衝擊，又高估之後每多一次的影響。
- 被當 2 次和 3 次的人很少，信賴區間很寬，而且和 1 次的區間大幅重疊，不能斷言 1、2、3 次之間有明確的高低順序。這次程式沒有出現 `FEW_EVENTS` 或 `SEPARATION` 警告。

### 其他變數

- higher：想讀大學的學生，及格勝算是不想讀的 3.500 倍（95% CI 1.872 – 6.475），預測機率 0.927 對 0.783。
- studytime：每多一級，及格勝算乘以 1.400（95% CI 1.011 – 1.980），預測機率從 0.902（等級 1）到 0.928（等級 2）。信賴區間下限很接近 1，證據比模型 A 更邊緣。
- sex：勝算比 0.793（95% CI 0.484 – 1.303），區間包含 1，沒有證據顯示男女有差異。

預測能力 AUC = 0.800，和模型 A（0.799）幾乎一樣。模型 B 的配適度比較好（前一次比較 P = 0.009），但區分及格與否的能力沒有明顯提升。

### 注意事項

- `GOF_NOTE`：studytime 仍然當數值，所以 deviance 不能當適合度檢定（§5.2.3）。
- 這是觀察資料，看到的是關聯，不是因果。
- 勝算比不是機率的倍數。例如勝算比 0.164 對應的是機率從 0.927 降到 0.674。

### 下一步

- 檢查 studytime：可以比較它當數值和當類別（§4.3）。不過等級 4 只有 35 人，當類別會比較不穩定。請回覆比較 studytime。
- 簡化 failures：結果顯示 1、2、3 次很接近，可以把 failures 改成「有沒有被當過」兩類，再用 AIC 比較。不過目前程式無法直接合併類別，需要先新增一個欄位。請回覆比較有無被當過。
- 更新報告和壓縮檔：把模型 B 加進 report.html 和 analysis_results.zip。請回覆更新報告。

---

## 7. 使用者

> 更新報告

## 7. Claude

更新 `summary.md`（說明模型比較的結果，並以模型 B 為最終模型），再執行 `report`，把 `output/q1`、`output/q1_compare`、`output/q1_cat` 三份結果整理進 `output/report.html`，重新打包 `analysis_results.zip`。
