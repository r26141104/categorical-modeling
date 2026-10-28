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
