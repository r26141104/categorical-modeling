# Skill 實測紀錄

每次實測都在 Claude App 的**新對話**中進行：只上傳 Skill 和資料（`student_por.csv`），像一般使用者一樣提問。數字另外用 R 獨立重算（`report/verify.R`）。

| | 第一次 `round1/` | 第二次 `round2/` | 第三次 `round3/` |
|---|---|---|---|
| 日期 | 2026/10/26 | 2026/10/28–29 | 2026/11/01 |
| 目的 | 三個模型是否選對、數字是否正確 | 新功能：模型比較、重新配適、產生報告 | 改版後的行為：協助判斷、接續分析、選擇題 |
| 二元 logistic（是否及格） | ✅ | ✅ | |
| 累積 logit（週末飲酒量） | ✅ | | ✅ |
| baseline-category logit（選校原因） | ✅ | | |
| 模型比較（failures 當數值 vs 類別） | | ✅ | |
| 用較好的模型重新配適 | | ✅ | |
| 產生 HTML 報告 | | ✅ | |
| 協助判斷、接續分析、不獨立就停止、選擇題 | | | ✅ |
| 和 R 驗證的比對 | 21 項全部一致 | 6 + 9 + 16 項全部一致 | 數字和第一次完全相同 |

三次實測測的是不同的部分，合起來才是完整的驗證。

## 每個資料夾的內容

```text
skill-test/
├── round1/
│   ├── q1/  q2/  q3/       三題的結果（result.json）與圖
│   ├── plots/              配適前畫的長條圖
│   ├── screenshots/        Skill 確認問題的截圖
│   └── transcript.md       對話紀錄
├── round2/
│   ├── q1/                 第 1 題重跑（failures 當數值）
│   ├── q1_compare/         模型比較的結果
│   ├── q1_cat/             failures 當類別重新配適
│   ├── report.html         Skill 產生的分析報告
│   ├── screenshots/        截圖
│   └── transcript.md       對話紀錄
└── round3/
    ├── record.md           測試紀錄：每個測試的提問、預期、結果、數字核對
    ├── screenshots/        截圖
    └── transcript.md       對話紀錄（依截圖整理）
```

## 資料格式轉換的驗證（`tidy-check/`）

課本的 Polviews 是交叉表（性別、政黨 × 五種意識形態，共 661 人）。用 `tidy --layout crosstab` 展開成每人一列（`polviews_tidy.csv`、`tidy_log.txt`），再用 Skill 配適累積 logit（`ordinal/result.json`）：共和黨 vs 民主黨的勝算比 37.85、比例勝算檢定 LR = 6.22（df = 6），和直接用交叉表在 R（`VGAM::vglm`）計算的結果相同。
