# categorical-modeling

一個 Claude Agent Skill：**類別結果變數的模型分析顧問**。

上傳資料、用平常說話的方式提問，Skill 會判斷該用哪一個模型、說明假設與限制，用內附的 Python 程式實際配適（**不需要網路、不需要安裝任何套件**），並以白話解釋勝算比與預測機率；還能**比較兩個模型**，最後把分析**整理成一份 HTML 報告**。

統計內容依據 Agresti, A. (2019). *An Introduction to Categorical Data Analysis* (3rd ed.) 第 4–6 章。

| 線上看（不用下載） | |
|---|---|
| 📘 [Part 2 報告](https://r26141104.github.io/categorical-modeling/report/modeling.html) | 方法、Skill 介紹（檔案結構與每個檔案的功能）、三次實測結果（**從這裡開始看**） |
| 🛠️ [製作教學](https://r26141104.github.io/categorical-modeling/make-skill-tutorial.html) | 規劃、撰寫、測試、修改的完整過程 |
| 🤖 [AI 使用紀錄](https://r26141104.github.io/categorical-modeling/report/ai-usage.html) | 製作過程中使用 AI 的紀錄 |

**想自己用：** 下載 [categorical-modeling.zip](https://github.com/r26141104/categorical-modeling/raw/main/categorical-modeling.zip)，上傳到 Claude 即可（步驟見下方「安裝」）。

## 涵蓋範圍

| 結果變數 | 模型 | 教科書 |
|---|---|---|
| 兩類 | 二元 logistic 迴歸 | §4.1–4.6、§5.1–5.3 |
| 三類以上、沒有順序 | baseline-category logit | §6.1 |
| 三類以上、有順序 | 累積 logit（比例勝算模型） | §6.2–6.3 |
| 上面任一種 | 比較兩個模型：概似比檢定、AIC | §4.3、§5.1–5.2 |

範圍外：配對／重複測量／叢集資料、計數結果、連續結果、只檢定兩變數關聯（不建模型）。遇到這些情況，Skill 會說明原因和適合的方法，不會硬套。

## 安裝

**Claude App（網頁版、桌面版）**

1. **Settings → Capabilities**：確認「程式碼執行與檔案建立」（Code execution and file creation）已開啟。**網路存取不用開**，Skill 不需要網路。
2. **Customize → Skills**：按 **＋** → **Create skill** → **Upload a skill**，上傳 `categorical-modeling.zip`
3. 確認 Skill 已開啟，開一個新對話，上傳資料檔後提問

Skill 只用 Claude 程式執行環境本來就有的 numpy、pandas（matplotlib 用來畫圖，沒有也能算）。模型配適是自己寫的程式（`scripts/catfit.py`），
**不需要 statsmodels、scipy，也不會執行 `pip install`**。每個對話第一次分析前，Skill 會先執行 `run_model.py check`，用內附資料和 R 的結果比對，確認這個環境算得正確。

**Claude Code**：把 Skill 資料夾複製到 `<你的專案>/.claude/skills/categorical-modeling/`，或個人層級的 `~/.claude/skills/categorical-modeling/`。

## 怎麼使用

直接用自然語言描述需求並上傳資料，不需要知道模型名稱。例如：

> 用 categorical-modeling skill 分析 crabs.csv：背甲寬度（width）和顏色（color）會影響身邊有沒有衛星雄蟹（y）嗎？

> alligators.csv 裡，鱷魚的體長（length）和主食（food）有關嗎？

> 我的結果是滿意度 1–5 分，該用哪個模型？（只問概念，不會開始配適）

### 分析流程

流程分成兩段：**A 選模型**決定用哪一個模型，三個模型都接到 **B 分析流程**（圖檔版：[`decision-tree.png`](decision-tree.png)）。

**A　選模型**

```mermaid
flowchart TD
 A0["資料檔＋研究問題"] --> A1["tidy：整理成每個觀察對象一列<br>（原本就是就跳過）"]
 A1 --> B["describe：程式自動算出<br>樣本數、變數型態、類別數、次數、缺失"]
 B --> C{{"① 觀察對象彼此獨立？<br>要問使用者｜§1.1.3"}}
 C -->|否| Z1["範圍外<br>McNemar、GEE、GLMM｜Ch.8–10"]
 C -->|是| E{{"結果 Y 的型態？"}}
 E -->|計數| Z2["範圍外<br>Poisson／負二項｜§3.3"]
 E -->|連續| Z3["範圍外<br>一般線性迴歸"]
 E -->|類別| F{{"Y 有幾類？<br>程式自動計算"}}
 F -->|兩類| M1["二元 logistic 迴歸<br>§4"]
 F -->|三類以上| G{{"② 類別有自然順序？<br>Skill 建議、使用者確認"}}
 G -->|沒有| M2["baseline-category logit<br>§6.1"]
 G -->|有| M3["累積 logit<br>§6.2"]

 classDef ask fill:#FFF4D6,stroke:#C99A2E,color:#3A2E00
 classDef model fill:#DCE8F7,stroke:#1F3A5F,stroke-width:2px,color:#10233F
 classDef stop fill:#EFEFEF,stroke:#888,color:#333
 class C,G ask
 class M1,M2,M3 model
 class Z1,Z2,Z3 stop
```

**B　分析流程**

```mermaid
flowchart TD
 K["確認清單 ①–⑥<br>選擇題：照預設配適（推薦）"]
 K -.->|先畫圖（可選）| P["plot：比例長條圖"]
 K --> L["配適模型<br>LR 檢定、勝算比、預測機率"]
 P -.-> L
 L --> W["檢查警告代碼<br>完全分離 §5.3、比例勝算 §6.3.2"]
 W --> R["白話解釋結果"]
 R --> Q{{"選擇題：下一步"}}
 Q -->|比較模型| T["compare<br>巢狀：LR 檢定 §5.2<br>非巢狀：AIC §5.1"]
 Q -->|產生報告| V["report<br>產生 HTML 報告"]

 classDef ask fill:#FFF4D6,stroke:#C99A2E,color:#3A2E00
 class K,Q ask
```

每個判斷點的理由、使用者不確定時怎麼協助判斷，見 [`references/decision-tree.md`](references/decision-tree.md)。

### 支援的資料格式

資料不一定要是「每個觀察對象一列」，Skill 會先用 `tidy` 自動整理，再請你確認轉換結果：

| 資料長相 | 例子 |
|---|---|
| 每個觀察對象一列（標準格式） | 每位學生一列，欄位是 sex、studytime、pass |
| 每個觀察對象一欄 | 第一欄是變數名稱，之後每一欄是一位學生 |
| 次數表 | 每一列是一種組合，另有一欄是人數（count） |
| 交叉表 | 列是解釋變數的類別，欄是結果的類別，格子裡是人數（例如課本的 Polviews） |

### 流程中 Claude 會問你什麼

描述資料後，Claude 會用**一則訊息**列出確認清單，每一項附預設值與理由：

| 編號 | 確認事項 | 說明 |
|---|---|---|
| ① | 每個觀察對象（例如每位學生）是否只出現一次、彼此獨立 | 資料本身看不出來，一定要你確認 |
| ② | 結果的類別數與順序 | 類別數由程式自動算出；Skill 依變數意義建議有沒有順序，你只要確認 |
| ③ | 數字代碼當數值分數還是類別 | 例如讀書時間 1–4 |
| ④ | 類別型解釋變數的參考組 | 預設排序後第一個類別 |
| ⑤ | 二元結果的成功類別、名目結果的基準類別 | 只改變勝算比的方向 |
| ⑥ | 樣本數是否足夠 | Claude 的檢查結果，不需要你決定 |

每則回覆的結尾都是**選擇題**，推薦的選項放在第一個並標「（推薦）」，推薦哪一個依固定規則決定（例如出現 `PO_VIOLATED` 時推薦「配適 multinomial 當對照」）。要改設定就選「其他」，寫編號和新設定，例如「③ failures 改當類別」。

### 三種工作模式

| 模式 | 說明 |
|---|---|
| 完整分析 | 從描述資料、確認清單一路做到解釋結果 |
| 只問概念或單一步驟 | 例如「勝算比怎麼解釋」，直接回答，不跑完整流程 |
| 接續分析 | 同一個對話已確認的設定直接沿用，不重複詢問 |

### Skill 的規則

- **未經同意不畫圖、不配適、不比較、不產生報告**；資料是否獨立一定讓使用者確認。
- **所有數字都由程式計算**，Claude 不自己計算、修改數字，也不從圖上估讀。
- **Claude 不可以臨時自己寫迴歸程式**，一律用 `scripts/` 裡的程式；也不會安裝任何套件。

## 實測與驗證

計算用內附的 Python 程式 [`scripts/catfit.py`](scripts/catfit.py)：二元 logistic、baseline-category logit、累積 logit 都用 Newton-Raphson 自己配適，只需要 numpy。
所有輸出（係數、標準誤、Wald 檢定、profile likelihood／Wald 信賴區間、deviance、概似比檢定、AIC、預測機率、AUC）都和 statsmodels 與 R（`glm`、`VGAM::vglm`、`ordinal::clm`、`pROC`）比對過：
**146 項、1571 個數字，差距都小於 1e-6**（[`tests/RESULTS.md`](tests/RESULTS.md)）。比對時刻意擋掉 statsmodels、scipy，證明 Skill 執行時不需要它們。
完全分離（§5.3）時，程式會偵測到並把無限大的估計值輸出成 `Inf`、標準誤與信賴區間輸出 `NA`，不會給看起來正常的數字。

實測的數字另外用 R（`glm`、`MASS`、`ordinal`、`nnet`）獨立重算（[`report/verify.R`](report/verify.R)）。

| | 第一次 | 第二次 | 第三次 |
|---|---|---|---|
| 測了什麼 | 三個模型（三題） | 模型比較、重新配適、產生報告 | 改版後的行為：協助判斷、接續分析、選擇題 |
| 結果 | 模型判斷 3 題全對；21 項數字和 R 一致 | 6 + 9 + 16 項數字和 R 一致 | 5 個測試全部符合預期 |

- 被當次數當類別明顯比當數值好（LR = 9.53，df = 2，P = 0.009），這是人為判斷沒料到、Skill 用檢定證實的結果。
- 實測時發現的問題（看圖估讀比例、回覆次數太多、無法檢查等距假設、重複選項被忽略）都已修正後再測。
- 每次實測的結果、截圖與對話紀錄：[`report/skill-test/`](report/skill-test/)

## 資料夾結構

```text
categorical-modeling/
│
│  ── Skill 本體（打包成 categorical-modeling.zip）──
├── SKILL.md                    主檔：觸發條件、行為規則、分析流程
├── references/
│   ├── decision-tree.md        選模型的決策樹、協助使用者判斷、配適後建議的比較
│   ├── methods.md              三個模型速查：H0、統計量、效果量、指令
│   └── textbook-methods.md     課本細節：假設、公式、注意事項（附節號）
├── scripts/
│   ├── run_model.py            指令：整理格式、描述、畫圖、配適、比較模型、產生報告、環境檢查
│   └── catfit.py               模型配適（Newton-Raphson、分離偵測、卡方與常態分布），只用 numpy
├── data/                       範例資料：馬蹄蟹、鱷魚、葡萄牙高中生
├── decision-tree.png           決策樹圖
├── categorical-modeling.zip    打包好的 Skill，可直接上傳到 Claude
│
│  ── 說明網頁 ──
├── make-skill-tutorial.qmd / .html  製作教學
├── _nav/                            各網頁上方的導覽列
│
│  ── 比對測試（只在本機執行，不放進 zip）──
├── tests/
│   ├── test_against_reference.py   和 statsmodels、R 比對每個數字（差距要小於 1e-6）
│   ├── run_without_statsmodels.py  擋掉 statsmodels、scipy 後執行 Skill（模擬 claude.ai）
│   ├── make_reference_R.R          產生 R 的參考值 reference_R.json
│   └── RESULTS.md                  比對結果
│
│  ── 報告與實測 ──
└── report/
    ├── modeling.qmd / .html    Part 2 報告
    ├── ai-usage.qmd / .html    AI 使用紀錄（原始紀錄：ai-log.csv）
    ├── judgment.yml            實測前依課本做的人為判斷
    ├── verify.R                用 R 獨立重算的驗證程式（結果：verify_results.json）
    ├── data/agresti/           報告中課本例題用的資料
    └── skill-test/             三次實測：round1、round2、round3
```

重新跑比對測試（需要 statsmodels、scipy、scikit-learn；R 的參考值已附在 `tests/reference_R.json`）：`python tests/test_against_reference.py`

報告與網頁都可以重新產生：在對應的資料夾執行 `quarto render 檔名.qmd`（Part 2 報告需要 R 與 VGAM、car、pROC、ggplot2、yaml、jsonlite 套件）。

<details>
<summary><b>進階：直接執行 run_model.py</b></summary>

只需要 Python 3 與 `numpy`、`pandas`（要畫圖再加 `matplotlib`）。不需要 statsmodels、scipy，也不需要網路。

```bash
# 檢查環境，並用內附資料和 R 的結果比對
python scripts/run_model.py check

# 整理資料格式（每個觀察對象一欄、次數表、交叉表 → 每個觀察對象一列）
python scripts/run_model.py tidy        --csv raw.csv --layout transpose --file tidy.csv
python scripts/run_model.py tidy        --csv raw.csv --layout freq --freq count --file tidy.csv
python scripts/run_model.py tidy        --csv raw.csv --layout crosstab --keep gender,party --y ideology --file tidy.csv

# 描述資料、畫圖
python scripts/run_model.py describe    --csv data.csv --vars y,a,b
python scripts/run_model.py plot        --csv data.csv --y y --x a

# 三個模型（結果寫到 --out 資料夾：result.json + 圖）
python scripts/run_model.py logistic    --csv data.csv --y pass --x a,b --success pass --numeric a
python scripts/run_model.py multinomial --csv data.csv --y reason --x a,b --baseline other
python scripts/run_model.py ordinal     --csv data.csv --y rating --x a --levels 1,2,3,4,5

# 比較兩個模型：模型 A 用 --x、--numeric，模型 B 用 --x2、--numeric2（沒給就沿用 A）
python scripts/run_model.py compare     --type logistic --csv data.csv --y pass --x a,b --numeric a,b --numeric2 b --success pass

# 產生報告：把多個結果資料夾整理成一份 HTML
python scripts/run_model.py report      --out output/q1,output/q1_compare --summary summary.md --file output/report.html
```

| 選項 | 意義 |
|---|---|
| `--layout` | `tidy` 的原始格式：`transpose`（每個對象一欄）、`freq`（次數表）、`crosstab`（交叉表） |
| `--freq`、`--keep` | `tidy` 的次數欄位；交叉表中屬於解釋變數的欄位（其他欄是結果的類別） |
| `--x` | 解釋變數，多個用逗號分隔 |
| `--numeric` | 要當數值分數的變數；沒列的文字或代碼變數都當類別 |
| `--ref` | 類別型解釋變數的參考類別，例如 `--ref sex=F,school=GP` |
| `--success` | 二元結果的成功類別（決定勝算比方向） |
| `--baseline` | 名目結果的基準類別 |
| `--levels` | 有序結果由低到高的順序 |
| `--out` | 輸出資料夾（預設 `output`）；`report` 時是要整理的結果資料夾（逗號分隔） |
| `--type` | `compare` 的模型種類：`logistic`、`multinomial`、`ordinal` |
| `--x2`、`--numeric2` | `compare` 的模型 B；`--numeric2 none` 代表全部當類別 |
| `--title`、`--summary`、`--file` | `report` 的標題、結論摘要檔（Markdown 文字）、輸出檔名 |

</details>

## 參考文獻

Agresti, A. (2019). *An Introduction to Categorical Data Analysis* (3rd ed.). Wiley.

Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

本 Skill 是「類別資料分析」課程報告 Part 2：Modeling Categorical Outcomes 的作品。
