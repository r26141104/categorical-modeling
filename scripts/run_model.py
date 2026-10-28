#!/usr/bin/env python3
"""run_model.py — 類別結果模型 Skill 的計算程式（Python 版）

依據：Agresti, A. (2019). An Introduction to Categorical Data Analysis (3rd ed.)
  二元 logistic §4.1–4.6、§5.1–5.3；baseline-category logit §6.1；累積 logit §6.2–6.3

用法：
  python scripts/run_model.py describe    --csv data.csv [--vars a,b,c]
  python scripts/run_model.py plot        --csv data.csv --y Y [--x X] [--out output]
  python scripts/run_model.py logistic    --csv data.csv --y Y --x a,b [--success 類別] [--numeric a] [--ref b=類別] [--out output]
  python scripts/run_model.py multinomial --csv data.csv --y Y --x a,b [--baseline 類別] [--numeric a] [--ref b=類別] [--out output]
  python scripts/run_model.py ordinal     --csv data.csv --y Y --x a,b [--levels 低,…,高] [--numeric a] [--ref b=類別] [--out output]
  python scripts/run_model.py compare     --type logistic --csv data.csv --y Y --x a,b --numeric a --x2 a,b --numeric2 none [--out output]
  python scripts/run_model.py report      --out output/q1,output/q2 [--title 標題] [--summary 摘要.md] [--file report.html]

每個分析都會在 --out 資料夾（預設 output/）寫出 result.json 與圖檔，並在畫面印出摘要。
需要的套件：numpy、pandas、scipy、statsmodels、matplotlib（缺少時：pip install statsmodels）
"""

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError:  # pragma: no cover
    sys.exit("需要 statsmodels 套件：請先執行 pip install statsmodels")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 圖表字型：有中文字型就用中文標籤，否則用英文

def _setup_font():
    wanted = ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft JhengHei", "Microsoft YaHei",
              "PingFang TC", "Heiti TC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Source Han Sans TC", "SimHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

ZH = _setup_font()
def L(zh, en):
    return zh if ZH else en

# ---------------------------------------------------------------- 共用工具

def fmt(x, k=3):
    return f"{x:.{k}f}"

def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "= NA"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

def sexp(v):
    """不會溢位的 exp：太大回傳無限大（例如完全分離時的勝算比）。"""
    v = float(v)
    if v > 700:
        return math.inf
    if v < -700:
        return 0.0
    return math.exp(v)

def split_arg(x):
    return [s.strip() for s in x.split(",") if s.strip()] if x else []

def read_any(path):
    if not os.path.exists(path):
        sys.exit(f"找不到資料檔：{path}")
    with open(path, "rb") as fh:
        head = fh.read(4096)
    sep = ";" if head.count(b";") > head.count(b",") else ","
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit("無法讀取資料檔的文字編碼（試過 UTF-8、Big5）。")

def need_cols(d, cols):
    miss = [c for c in cols if c not in d.columns]
    if miss:
        sys.exit(f"資料中沒有這些欄位：{'、'.join(miss)}。可用的欄位：{'、'.join(map(str, d.columns))}")

def as_str(v):
    """把類別值轉成字串；整數值的浮點數（例如 1.0）寫成 1。"""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)

def natural_levels(values):
    """類別排序：全部是數字就照數值排，否則照字母排（和 R 的 factor 相同）。"""
    u = sorted(set(values))
    try:
        return sorted(u, key=lambda s: float(s))
    except ValueError:
        return sorted(u)

def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "Inf" if f > 0 else "-Inf"
        return f
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

def write_result(res, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(res), fh, ensure_ascii=False, indent=2)
    print(f"\n結果已存到：{os.path.abspath(path)}")

def savefig(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name

# ---------------------------------------------------------------- 解釋變數的設計矩陣

class Design:
    """--numeric 指定的變數當數值；其他變數當類別，第一個類別（或 --ref 指定的）為參考組。
    類別變數的指標欄命名和 R 相同：變數名 + 類別，例如 sexM、higheryes。"""

    def __init__(self, d, x, numeric_vars, ref):
        self.x, self.numeric = x, set(numeric_vars)
        refs = {}
        for r in split_arg(ref):
            k, _, v = r.partition("=")
            refs[k.strip()] = v.strip()
        self.levels, self.cols = {}, {}
        for v in x:
            if v in self.numeric:
                vals = pd.to_numeric(d[v], errors="coerce")
                if vals.isna().all():
                    sys.exit(f"{v} 無法轉成數值。")
                self.cols[v] = [v]
            else:
                lv = natural_levels([as_str(s) for s in d[v]])
                if v in refs:
                    if refs[v] not in lv:
                        sys.exit(f"--ref：{refs[v]} 不是 {v} 的類別（{'、'.join(lv)}）。")
                    lv = [refs[v]] + [s for s in lv if s != refs[v]]
                self.levels[v] = lv
                self.cols[v] = [f"{v}{s}" for s in lv[1:]]

    def matrix(self, d, drop=None, const=True):
        parts = []
        for v in self.x:
            if v == drop:
                continue
            if v in self.numeric:
                parts.append(pd.DataFrame({v: pd.to_numeric(d[v], errors="coerce").astype(float).values}))
            else:
                s = d[v].map(as_str)
                parts.append(pd.DataFrame({f"{v}{lv}": (s == lv).astype(float).values for lv in self.levels[v][1:]}))
        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(d)))
        if const:
            X.insert(0, "(Intercept)", 1.0)
        return X

    def typical(self, d):
        """其他變數固定在平均數（數值）或最常見類別（類別）。"""
        t = {}
        for v in self.x:
            t[v] = float(pd.to_numeric(d[v]).mean()) if v in self.numeric else d[v].map(as_str).value_counts().idxmax()
        return t

    def vary(self, d, v):
        """要比較的值：數值變數用第 1、3 四分位數（兩者相同時用最小、最大值），類別變數用每個類別。"""
        if v in self.numeric:
            s = pd.to_numeric(d[v])
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            return ([float(s.min()), float(s.max())], ["最小值", "最大值"]) if q1 == q3 else ([float(q1), float(q3)], ["Q1", "Q3"])
        return self.levels[v], self.levels[v]

    def newdata(self, d, v, vals):
        t = self.typical(d)
        nd = pd.DataFrame([t] * len(vals))
        nd[v] = vals
        return nd

    def label(self, term):
        t = term.split(":")[0]
        if t in self.numeric:
            return f"{t}（每增加 1）"
        for v in sorted(self.levels, key=len, reverse=True):
            if t.startswith(v) and t != v:
                return f"{v} = {t[len(v):]}（相對於 {self.levels[v][0]}）"
        return t

def drop_missing(d, cols, warns):
    ok = d[cols].notna().all(axis=1)
    if (~ok).any():
        warns.append({"code": "MISSING", "message": f"有 {int((~ok).sum())} 筆資料因缺失值被排除；只有缺失完全隨機時才不會造成偏誤。"})
    return d.loc[ok].reset_index(drop=True)

# ---------------------------------------------------------------- describe

def cmd_describe(a):
    d = read_any(a.csv)
    vars_ = split_arg(a.vars) or list(d.columns)
    need_cols(d, vars_)
    print(f"資料檔：{a.csv}；{len(d)} 列 × {d.shape[1]} 欄")
    for v in vars_:
        x = d[v]
        u = x.dropna().unique()
        k = len(u)
        num = pd.api.types.is_numeric_dtype(x)
        if k == 2:
            guess = "二元（binary）"
        elif num and np.all(np.mod(x.dropna(), 1) == 0) and k <= 10:
            guess = "數字代碼：可能是有序類別、名目類別或計數，需確認"
        elif num:
            guess = "數值（連續或計數）"
        elif k <= 15:
            guess = "文字類別：名目或有序，需確認"
        else:
            guess = "文字，類別很多（可能是代號）"
        print(f"\n- {v}：{guess}；不同值 {k} 個；缺失 {int(x.isna().sum())}")
        if k <= 15:
            vc = x.map(lambda s: as_str(s) if pd.notna(s) else "NA").value_counts()
            vc = vc.reindex(natural_levels([s for s in vc.index if s != "NA"]) + (["NA"] if "NA" in vc.index else []))
            print("  次數：" + "、".join(f"{i} ({n})" for i, n in vc.items()))
        elif num:
            print(f"  範圍 {x.min():.2f} – {x.max():.2f}，平均 {x.mean():.2f}")
    hints = []
    for v in d.columns:
        name = str(v).lower()
        if (name == "id" or name.endswith("_id") or name.startswith("id_") or any(s in str(v) for s in ("編號", "代號", "學號"))) and d[v].duplicated().any():
            hints.append(f"欄位 {v} 有重複值：同一個體可能有多筆資料（配對或重複測量），獨立性的模型不適用。")
    print("\n資料結構提示：" + (" ".join(hints) if hints else "沒有發現重複的代號欄位；但每一列是否為不同、獨立的個體，仍須向使用者確認。"))

# ---------------------------------------------------------------- plot

def cmd_plot(a):
    d = read_any(a.csv)
    out = a.out or "output"
    need_cols(d, [a.y] + ([a.x] if a.x else []))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if not a.x:
        vc = d[a.y].map(as_str).value_counts()
        vc = vc.reindex(natural_levels(list(vc.index)))
        print(f"{a.y} 的分布：" + "、".join(f"{k} {n}（{n / vc.sum():.1%}）" for k, n in vc.items()))
        ax.bar(vc.index, vc.values, color="#1f3a5f")
        ax.set_xlabel(a.y); ax.set_ylabel(L("人數", "Count"))
        name = f"dist_{a.y}.png"
    else:
        xv = d[a.x]
        if pd.api.types.is_numeric_dtype(xv) and xv.nunique() > 10:
            xv = pd.qcut(xv, 4, duplicates="drop").astype(str)
        cnt = pd.crosstab(xv.map(as_str), d[a.y].map(as_str))
        cnt = cnt.reindex(natural_levels(list(cnt.index)))[natural_levels(list(cnt.columns))]
        tab = cnt.div(cnt.sum(axis=1), axis=0)
        print(f"{a.y} 依 {a.x} 分組的比例（每一列合計 100%；圖上的數字一律以這張表為準）")
        for g in tab.index:
            print(f"- {a.x} = {g}（n = {int(cnt.loc[g].sum())}）：" + "、".join(f"{c} {tab.loc[g, c]:.1%}" for c in tab.columns))
        bottom = np.zeros(len(tab))
        for c in tab.columns:
            ax.bar(tab.index, tab[c].values, bottom=bottom, label=c, edgecolor="white")
            bottom += tab[c].values
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel(a.x); ax.set_ylabel(L("比例", "Proportion")); ax.legend(title=a.y, bbox_to_anchor=(1.02, 1), loc="upper left")
        name = f"bar_{a.y}_by_{a.x}.png"
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out, name)
    print(f"圖已存到：{os.path.abspath(os.path.join(out, name))}")

# ---------------------------------------------------------------- 二元 logistic（§4.1–4.6、§5.1–5.3）

def auc_value(y, p):
    """ROC 曲線下面積 = concordance index（§4.6.2），以等級計算，同分算 1/2。"""
    r = stats.rankdata(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fit_logit(y, X, offset=None):
    return sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()

def profile_ci(y, X, fit, level=0.95):
    """profile likelihood 信賴區間（§4.2.3）：找使 2[ℓ(β̂) − ℓ_profile(b)] = χ²₁ 臨界值的 b。"""
    crit = stats.chi2.ppf(level, 1)
    llmax = fit.llf
    out = np.full((X.shape[1], 2), np.nan)
    for j, col in enumerate(X.columns):
        b, se = fit.params[col], fit.bse[col]
        others = X.drop(columns=col)
        def g(val):
            off = X[col].values * val
            if others.shape[1] == 0:
                p = 1 / (1 + np.exp(-off))
                ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            else:
                ll = fit_logit(y, others, offset=off).llf
            return 2 * (llmax - ll) - crit
        for side, sgn in ((0, -1), (1, 1)):
            step = max(se, 1e-3)
            lo, hi = b, b + sgn * step
            k = 0
            while g(hi) < 0 and k < 60:
                lo, hi = hi, hi + sgn * step
                step *= 1.6
                k += 1
            out[j, side] = optimize.brentq(g, min(lo, hi), max(lo, hi)) if k < 60 else (np.inf * sgn)
    return out

def cmd_logistic(a):
    d = read_any(a.csv)
    out = a.out or "output"
    x = split_arg(a.x)
    need_cols(d, [a.y] + x)
    warns = []
    d = drop_missing(d, [a.y] + x, warns)
    ys = d[a.y].map(as_str)
    lv = natural_levels(list(ys.unique()))
    if len(lv) != 2:
        sys.exit(f"{a.y} 有 {len(lv)} 類；二元 logistic 需要剛好 2 類（多類別請用 multinomial 或 ordinal）。")
    success = a.success if a.success else lv[1]
    if success not in lv:
        sys.exit(f"--success 必須是 {a.y} 的類別之一：{'、'.join(lv)}")
    y = (ys == success).astype(float).values
    des = Design(d, x, split_arg(a.numeric), a.ref)
    X = des.matrix(d)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_logit(y, X)
    sm_sep = any("Separation" in type(w.message).__name__ or "separation" in str(w.message).lower() for w in caught)
    fit0 = fit_logit(y, X[["(Intercept)"]])
    params, bse = fit.params, fit.bse

    sep_terms = [t for t in X.columns if abs(params[t]) > 10 and bse[t] > 50]
    if sm_sep and not sep_terms:
        sep_terms = [t for t in X.columns if t != "(Intercept)"]
    if sep_terms:
        warns.append({"code": "SEPARATION", "message": f"可能有完全或準完全分離：{'、'.join(sep_terms)} 的估計值其實是無限大，"
                      "Wald 檢定與標準誤沒有意義，請改看概似比檢定（§5.3）。"})
    ci_type = "profile likelihood"
    try:
        if sep_terms:
            raise ValueError
        ci = profile_ci(y, X, fit)
    except Exception:
        ci_type = "Wald"
        ci = np.column_stack([params - 1.96 * bse, params + 1.96 * bse])
    coefs = [{"term": t, "label": des.label(t) if t != "(Intercept)" else "(Intercept)",
              "estimate": params[t], "se": bse[t], "z": fit.tvalues[t], "p": fit.pvalues[t],
              "or": sexp(params[t]), "or_lower": sexp(ci[i, 0]), "or_upper": sexp(ci[i, 1])}
             for i, t in enumerate(X.columns)]

    lr = []
    for v in x:
        f0 = fit_logit(y, des.matrix(d, drop=v))
        s = f0.deviance - fit.deviance
        df = len(des.cols[v])
        lr.append({"variable": v, "lr": s, "df": df, "p": stats.chi2.sf(s, df)})
    ov_lr, ov_df = fit0.deviance - fit.deviance, X.shape[1] - 1
    overall = {"lr": ov_lr, "df": ov_df, "p": stats.chi2.sf(ov_lr, ov_df)}

    phat = np.asarray(fit.predict(X))
    auc = auc_value(y, phat)
    n_min, n_par = int(min(y.sum(), len(y) - y.sum())), X.shape[1] - 1
    if n_min / max(1, n_par) < 10:
        warns.append({"code": "FEW_EVENTS", "message": f"較少的那一類只有 {n_min} 個，模型有 {n_par} 個參數（每個參數不到 10 個），估計可能不穩定（§5.1.1）。"})
    if des.numeric:
        warns.append({"code": "GOF_NOTE", "message": "模型含數值解釋變數，deviance 不能當適合度檢定；要檢查模型，請和更複雜的模型（例如加交互作用）做概似比比較（§5.2.3）。"})

    effects = []
    for v in x:
        vals, names = des.vary(d, v)
        pr = fit.predict(des.matrix(des.newdata(d, v, vals)))
        for nm, val, p in zip(names, vals, pr):
            effects.append({"variable": v, "value": nm, "at": as_str(val), "probability": float(p)})

    plots = []
    num = [v for v in x if v in des.numeric]
    if num:
        v = num[0]
        s = pd.to_numeric(d[v])
        g = np.linspace(s.min(), s.max(), 100)
        Xg = des.matrix(des.newdata(d, v, list(g)))
        eta = Xg.values @ params.values
        se_eta = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, fit.cov_params().values, Xg.values))
        expit = lambda z: 1 / (1 + np.exp(-z))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.fill_between(g, expit(eta - 1.96 * se_eta), expit(eta + 1.96 * se_eta), color="#1f3a5f", alpha=0.15)
        ax.plot(g, expit(eta), color="#1f3a5f", lw=2)
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel(v); ax.set_ylabel(L(f"預測「{success}」的機率", f"P({a.y} = {success})"))
        ax.set_title(L("陰影：95% 信賴帶；其他變數固定在平均數或最常見類別", "Band: 95% CI; other variables at mean / mode"), fontsize=9, loc="right")
        ax.spines[["top", "right"]].set_visible(False)
        plots.append(savefig(fig, out, "probability_curve.png"))
    o = np.argsort(-phat, kind="mergesort")
    yy = y[o]
    fpr = np.concatenate([[0], np.cumsum(1 - yy) / (1 - yy).sum()])
    tpr = np.concatenate([[0], np.cumsum(yy) / yy.sum()])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.step(fpr, tpr, color="#b2182b", lw=2, where="post")
    ax.text(0.55, 0.15, f"AUC = {auc:.3f}", fontsize=13)
    ax.set_xlabel(L("1 − 特異度", "1 − specificity")); ax.set_ylabel(L("敏感度", "Sensitivity")); ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    plots.append(savefig(fig, out, "roc.png"))

    print(f"【二元 logistic 迴歸】結果變數 {a.y}，成功類別 = {success}；n = {len(y)}（成功 {int(y.sum())}）")
    print(f"整體檢定（所有解釋變數 vs 只有截距）：LR = {fmt(ov_lr, 2)}，df = {ov_df}，P {fmt_p(overall['p'])}")
    print(f"\n係數與勝算比（信賴區間：{ci_type}）")
    for c in coefs[1:]:
        print(f"- {c['label']}：β = {fmt(c['estimate'])}（SE {fmt(c['se'])}），勝算比 = {fmt(c['or'])}，"
              f"95% CI {fmt(c['or_lower'])} – {fmt(c['or_upper'])}，Wald P {fmt_p(c['p'])}")
    print("\n每個變數的概似比檢定（拿掉該變數）")
    for r in lr:
        print(f"- {r['variable']}：LR = {fmt(r['lr'], 2)}，df = {r['df']}，P {fmt_p(r['p'])}")
    print(f"\n預測能力：AUC = {fmt(auc)}")
    print("預測機率（其他變數固定在平均數或最常見類別）")
    for e in effects:
        print(f"- {e['variable']} = {e['value']}（{e['at']}）：{fmt(e['probability'])}")
    if warns:
        print("\n注意事項")
        for w in warns:
            print(f"- [{w['code']}] {w['message']}")

    write_result({"model": "logistic", "data": a.csv, "y": a.y, "success": success, "x": x,
                  "numeric": split_arg(a.numeric), "n": len(y), "n_success": int(y.sum()),
                  "reference": {v: des.levels[v][0] for v in des.levels}, "ci_type": ci_type,
                  "coefficients": coefs, "lr_tests": lr, "overall": overall, "auc": auc,
                  "effects": effects, "warnings": warns, "plots": plots}, out)

# ---------------------------------------------------------------- 多類別模型共用

def plot_categories(eff, v, cats, ordered, out, ylab):
    rows = [e for e in eff if e["variable"] == v]
    groups = [r["value"] for r in rows]
    if ordered:
        cmap = plt.get_cmap("Blues")
        colors = [cmap(0.95 - 0.75 * i / max(1, len(cats) - 1)) for i in range(len(cats))]
    else:
        colors = plt.get_cmap("tab10").colors[:len(cats)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = np.zeros(len(rows))
    for k, c in zip(cats, colors):
        vals = np.array([r["probs"][k] for r in rows])
        ax.bar(groups, vals, bottom=bottom, color=c, edgecolor="white", width=0.6, label=k)
        for i, (b, h) in enumerate(zip(bottom, vals)):
            if h >= 0.05:
                ax.text(i, b + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if (ordered and cats.index(k) < len(cats) / 2) else "black")
        bottom += vals
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(v); ax.set_ylabel(L("預測機率", "Predicted probability"))
    ax.set_title(L("其他變數固定在平均數或最常見類別", "Other variables at mean / mode"), fontsize=9, loc="right")
    ax.legend(title=ylab, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return savefig(fig, out, "category_probabilities.png")

def print_effects(eff, cats):
    print("\n預測機率（其他變數固定）")
    for e in eff:
        print(f"- {e['variable']} = {e['value']}：" + "、".join(f"{k} {fmt(e['probs'][k])}" for k in cats))

# ---------------------------------------------------------------- baseline-category logit（§6.1）

def cmd_multinomial(a):
    d = read_any(a.csv)
    out = a.out or "output"
    x = split_arg(a.x)
    need_cols(d, [a.y] + x)
    warns = []
    d = drop_missing(d, [a.y] + x, warns)
    ys = d[a.y].map(as_str)
    lv = natural_levels(list(ys.unique()))
    if len(lv) < 3:
        sys.exit(f"{a.y} 只有 {len(lv)} 類，請用 logistic。")
    base = a.baseline if a.baseline else lv[-1]
    if base not in lv:
        sys.exit(f"--baseline 必須是 {a.y} 的類別之一：{'、'.join(lv)}")
    order = [base] + [k for k in lv if k != base]          # statsmodels 以第一類（代碼 0）為基準
    yc = ys.map({k: i for i, k in enumerate(order)}).values
    des = Design(d, x, split_arg(a.numeric), a.ref)
    X = des.matrix(d)
    fit = sm.MNLogit(yc, X).fit(method="newton", maxiter=200, disp=False)
    fit0 = sm.MNLogit(yc, X[["(Intercept)"]]).fit(method="newton", maxiter=200, disp=False)
    others = order[1:]
    coefs = []
    for j, k in enumerate(others):                          # 依 R／VGAM 的慣例，以結果類別的字母順序編號方程式
        eqn = [o for o in lv if o != base].index(k) + 1
        for t in X.columns:
            b, se = fit.params.iloc[X.columns.get_loc(t), j], fit.bse.iloc[X.columns.get_loc(t), j]
            coefs.append({"term": f"{t}:{eqn}", "equation": f"{k} vs {base}",
                          "label": des.label(t) if t != "(Intercept)" else "(Intercept)",
                          "estimate": b, "se": se, "z": b / se, "p": 2 * stats.norm.sf(abs(b / se)),
                          "or": sexp(b), "or_lower": sexp(b - 1.96 * se), "or_upper": sexp(b + 1.96 * se)})
    ov_lr = 2 * (fit.llf - fit0.llf)
    ov_df = (X.shape[1] - 1) * (len(lv) - 1)
    overall = {"lr": ov_lr, "df": ov_df, "p": stats.chi2.sf(ov_lr, ov_df)}
    lr = []
    for v in x:
        Xr = des.matrix(d, drop=v)
        fr = sm.MNLogit(yc, Xr).fit(method="newton", maxiter=200, disp=False)
        s = 2 * (fit.llf - fr.llf)
        df = len(des.cols[v]) * (len(lv) - 1)
        lr.append({"variable": v, "lr": s, "df": df, "p": stats.chi2.sf(s, df)})
    if any(abs(c["estimate"]) > 10 and c["se"] > 50 for c in coefs):
        warns.append({"code": "SEPARATION", "message": "部分係數與標準誤異常大，可能有分離現象（某些組合從未出現某個類別），該係數的 Wald 檢定沒有意義（§5.3）。"})
    counts = ys.value_counts()
    if counts.min() < 10:
        warns.append({"code": "FEW_EVENTS", "message": "有類別的觀察值少於 10 個，估計可能不穩定。"})
    eff = []
    for v in x:
        vals, names = des.vary(d, v)
        pr = np.asarray(fit.predict(des.matrix(des.newdata(d, v, vals))))
        for i, nm in enumerate(names):
            eff.append({"variable": v, "value": nm, "at": as_str(vals[i]), "probs": {k: float(pr[i, order.index(k)]) for k in lv}})
    plots = [plot_categories(eff, x[0], lv, False, out, a.y)]

    print(f"【baseline-category logit】結果變數 {a.y}（" + "、".join(f"{k} {int(counts[k])}" for k in lv) + f"），基準類別 = {base}；n = {len(yc)}")
    print(f"整體檢定：LR = {fmt(ov_lr, 2)}，df = {ov_df}，P {fmt_p(overall['p'])}")
    print("\n各方程式的係數與勝算比（95% Wald 信賴區間）")
    for c in sorted([c for c in coefs if c["label"] != "(Intercept)"], key=lambda c: (c["label"], c["equation"])):
        print(f"- {c['equation']}｜{c['label']}：β = {fmt(c['estimate'])}，勝算比 = {fmt(c['or'])}（{fmt(c['or_lower'])} – {fmt(c['or_upper'])}），P {fmt_p(c['p'])}")
    print("\n每個變數的概似比檢定")
    for r in lr:
        print(f"- {r['variable']}：LR = {fmt(r['lr'], 2)}，df = {r['df']}，P {fmt_p(r['p'])}")
    print_effects(eff, lv)
    if warns:
        print("\n注意事項")
        for w in warns:
            print(f"- [{w['code']}] {w['message']}")
    eff_out = [{"variable": e["variable"], "value": e["value"], "at": e["at"], **e["probs"]} for e in eff]
    write_result({"model": "multinomial", "data": a.csv, "y": a.y, "categories": lv, "baseline": base, "x": x,
                  "numeric": split_arg(a.numeric), "n": len(yc), "coefficients": coefs, "lr_tests": lr,
                  "overall": overall, "effects": eff_out, "warnings": warns, "plots": plots}, out)

# ---------------------------------------------------------------- 累積 logit（§6.2–6.3）

def ordered_fit(yc, X):
    return OrderedModel(yc, X, distr="logit").fit(method="bfgs", maxiter=2000, disp=False)

def null_loglik(yc):
    n = np.bincount(yc)
    n = n[n > 0]
    return float(np.sum(n * np.log(n / n.sum())))

def general_cumlogit_loglik(yc, X, J, start_theta, start_beta):
    """非比例勝算的累積 logit：logit P(Y ≤ j) = θ_j − x'β_j（每個切點各有斜率），用來檢定比例勝算假設（§6.3.2）。"""
    Xv = X.values
    p = Xv.shape[1]
    def nll(par):
        th = par[:J - 1]
        B = par[J - 1:].reshape(J - 1, p)
        eta = th[None, :] - Xv @ B.T                         # n × (J−1)
        cum = 1 / (1 + np.exp(-eta))
        cum = np.column_stack([np.zeros(len(yc)), cum, np.ones(len(yc))])
        pr = cum[np.arange(len(yc)), yc + 1] - cum[np.arange(len(yc)), yc]
        if np.any(pr <= 0):
            return 1e10 + 1e6 * np.sum(np.clip(-pr, 0, None))
        return -np.sum(np.log(pr))
    x0 = np.concatenate([start_theta, np.tile(start_beta, J - 1)])
    res = optimize.minimize(nll, x0, method="BFGS", options={"maxiter": 5000, "gtol": 1e-8})
    res = optimize.minimize(nll, res.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-10}) if not res.success else res
    return -res.fun

def cmd_ordinal(a):
    d = read_any(a.csv)
    out = a.out or "output"
    x = split_arg(a.x)
    need_cols(d, [a.y] + x)
    warns = []
    d = drop_missing(d, [a.y] + x, warns)
    ys = d[a.y].map(as_str)
    lv = split_arg(a.levels) if a.levels else natural_levels(list(ys.unique()))
    bad = sorted(set(ys) - set(lv))
    if bad:
        sys.exit(f"--levels 少了這些類別：{'、'.join(bad)}")
    if len(lv) < 3:
        sys.exit(f"{a.y} 只有 {len(lv)} 類，請用 logistic。")
    J = len(lv)
    yc = ys.map({k: i for i, k in enumerate(lv)}).values.astype(int)
    des = Design(d, x, split_arg(a.numeric), a.ref)
    X = des.matrix(d, const=False)
    fit = ordered_fit(yc, X)
    k = X.shape[1]
    beta, se = fit.params.iloc[:k], fit.bse.iloc[:k]
    # statsmodels：P(Y ≤ j) = F(θ_j − x'β)，β > 0 代表傾向較高類別；「落在較高類別」的勝算比 = exp(β)
    coefs = [{"term": t, "label": des.label(t), "estimate": beta[t], "se": se[t], "z": beta[t] / se[t],
              "p": 2 * stats.norm.sf(abs(beta[t] / se[t])), "or_higher": sexp(beta[t]),
              "or_lower": sexp(beta[t] - 1.96 * se[t]), "or_upper": sexp(beta[t] + 1.96 * se[t])} for t in X.columns]
    ll0 = null_loglik(yc)
    ov_lr, ov_df = 2 * (fit.llf - ll0), k
    overall = {"lr": ov_lr, "df": ov_df, "p": stats.chi2.sf(ov_lr, ov_df)}
    lr = []
    for v in x:
        Xr = des.matrix(d, drop=v, const=False)
        llr = ordered_fit(yc, Xr).llf if Xr.shape[1] > 0 else ll0
        s = 2 * (fit.llf - llr)
        df = len(des.cols[v])
        lr.append({"variable": v, "lr": s, "df": df, "p": stats.chi2.sf(s, df)})
    theta = fit.model.transform_threshold_params(fit.params.values)[1:-1]
    po = None
    try:
        ll_gen = general_cumlogit_loglik(yc, X, J, theta, beta.values)
        s = 2 * (ll_gen - fit.llf)
        df = (J - 2) * k
        po = {"lr": max(s, 0.0), "df": df, "p": stats.chi2.sf(max(s, 0.0), df)}
    except Exception:
        warns.append({"code": "PO_CHECK_FAILED", "message": "非比例勝算模型無法配適，無法檢定比例勝算假設（§6.3.2）。"})
    if po and po["p"] < 0.05:
        warns.append({"code": "PO_VIOLATED", "message": f"比例勝算假設的檢定 P {fmt_p(po['p'])}：各切點的效果可能不同（例如某組只是分布較分散，"
                      "而不是整體往上移），單一勝算比可能過度簡化，應分開描述（§6.3.2）。"})
    eff = []
    for v in x:
        vals, names = des.vary(d, v)
        Xn = des.matrix(des.newdata(d, v, vals), const=False)
        pr = np.asarray(fit.model.predict(fit.params.values, exog=Xn.values))
        for i, nm in enumerate(names):
            eff.append({"variable": v, "value": nm, "at": as_str(vals[i]), "probs": {kk: float(pr[i, j]) for j, kk in enumerate(lv)}})
    plots = [plot_categories(eff, x[0], lv, True, out, a.y)]
    counts = ys.value_counts()

    print(f"【累積 logit（比例勝算）】結果變數 {a.y}：" + " < ".join(f"{kk} ({int(counts.get(kk, 0))})" for kk in lv) + f"；n = {len(yc)}")
    print(f"整體檢定：LR = {fmt(ov_lr, 2)}，df = {ov_df}，P {fmt_p(overall['p'])}")
    print("\n係數與「落在較高類別」的勝算比（95% Wald 信賴區間；參數化 logit P(Y≤j) = θ_j − βx，β > 0 代表傾向較高類別）")
    for c in coefs:
        print(f"- {c['label']}：β = {fmt(c['estimate'])}，較高類別的勝算比 = {fmt(c['or_higher'])}（{fmt(c['or_lower'])} – {fmt(c['or_upper'])}），P {fmt_p(c['p'])}")
    print("\n每個變數的概似比檢定")
    for r in lr:
        print(f"- {r['variable']}：LR = {fmt(r['lr'], 2)}，df = {r['df']}，P {fmt_p(r['p'])}")
    if po:
        print(f"\n比例勝算假設檢定：LR = {fmt(po['lr'], 2)}，df = {po['df']}，P {fmt_p(po['p'])}（P 小表示假設可能不成立）")
    print_effects(eff, lv)
    if warns:
        print("\n注意事項")
        for w in warns:
            print(f"- [{w['code']}] {w['message']}")
    eff_out = [{"variable": e["variable"], "value": e["value"], "at": e["at"], **e["probs"]} for e in eff]
    write_result({"model": "ordinal", "data": a.csv, "y": a.y, "levels": lv, "x": x, "numeric": split_arg(a.numeric),
                  "parameterization": "logit P(Y<=j) = theta_j - x*beta（beta > 0 代表傾向較高類別）",
                  "n": len(yc), "coefficients": coefs, "lr_tests": lr, "overall": overall, "proportional_odds": po,
                  "effects": eff_out, "warnings": warns, "plots": plots}, out)

# ---------------------------------------------------------------- compare：比較兩個模型（§4.3、§5.1–5.2）
# 模型 A 用 --x、--numeric；模型 B 用 --x2、--numeric2（沒給就沿用 A 的設定；--numeric2 none 代表全部當類別）

MODEL_NAMES = {"logistic": "二元 logistic 迴歸", "multinomial": "baseline-category logit", "ordinal": "累積 logit（比例勝算）"}

def fit_loglik(kind, d, ys, a, x, numeric):
    """配適指定的模型，回傳（對數概似、參數個數）。"""
    des = Design(d, x, numeric, a.ref)
    if kind == "logistic":
        lv = natural_levels(list(ys.unique()))
        if len(lv) != 2:
            sys.exit(f"{a.y} 有 {len(lv)} 類；logistic 需要剛好 2 類。")
        success = a.success if a.success else lv[1]
        if success not in lv:
            sys.exit(f"--success 必須是 {a.y} 的類別之一：{'、'.join(lv)}")
        y = (ys == success).astype(float).values
        X = des.matrix(d)
        return fit_logit(y, X).llf, X.shape[1], des
    if kind == "multinomial":
        lv = natural_levels(list(ys.unique()))
        base = a.baseline if a.baseline else lv[-1]
        if base not in lv:
            sys.exit(f"--baseline 必須是 {a.y} 的類別之一：{'、'.join(lv)}")
        order = [base] + [k for k in lv if k != base]
        yc = ys.map({k: i for i, k in enumerate(order)}).values
        X = des.matrix(d)
        f = sm.MNLogit(yc, X).fit(method="newton", maxiter=200, disp=False)
        return f.llf, X.shape[1] * (len(lv) - 1), des
    lv = split_arg(a.levels) if a.levels else natural_levels(list(ys.unique()))
    bad = sorted(set(ys) - set(lv))
    if bad:
        sys.exit(f"--levels 少了這些類別：{'、'.join(bad)}")
    yc = ys.map({k: i for i, k in enumerate(lv)}).values.astype(int)
    X = des.matrix(d, const=False)
    ll = ordered_fit(yc, X).llf if X.shape[1] > 0 else null_loglik(yc)
    return ll, X.shape[1] + len(lv) - 1, des

def is_nested(small, big):
    """small 的每個變數都在 big 裡；small 當類別的變數，在 big 也必須當類別。
    （數值分數是類別編碼的特例，所以「數值 → 類別」是巢狀的，§4.3。）"""
    for v in small["x"]:
        if v not in big["x"]:
            return False
        if v not in small["numeric"] and v in big["numeric"]:
            return False
    return True

def cmd_compare(a):
    kind = a.type
    if kind not in MODEL_NAMES:
        sys.exit("請用 --type 指定模型：logistic、multinomial 或 ordinal。")
    d = read_any(a.csv)
    out = a.out or "output"
    xa, xb = split_arg(a.x), split_arg(a.x2) or split_arg(a.x)
    none = lambda v: [] if v is None or v.strip().lower() in ("", "none", "無") else split_arg(v)
    na = none(a.numeric)
    nb = na if a.numeric2 is None else none(a.numeric2)
    need_cols(d, [a.y] + xa + xb)
    warns = []
    d = drop_missing(d, [a.y] + sorted(set(xa) | set(xb)), warns)   # 兩個模型用同一批資料
    ys = d[a.y].map(as_str)
    models = []
    for name, x, num in (("A", xa, na), ("B", xb, [v for v in nb])):
        num = [v for v in num if v in x]
        ll, k, des = fit_loglik(kind, d, ys, a, x, num)
        models.append({"name": name, "x": x, "numeric": num,
                       "categorical": [v for v in x if v not in num],
                       "loglik": ll, "n_params": k, "deviance": -2 * ll, "aic": -2 * ll + 2 * k})
    A, B = models
    small, big = (A, B) if A["n_params"] <= B["n_params"] else (B, A)
    nested = small["n_params"] < big["n_params"] and is_nested(small, big)
    test = None
    if nested:
        s = max(2 * (big["loglik"] - small["loglik"]), 0.0)
        df = big["n_params"] - small["n_params"]
        test = {"simple": small["name"], "complex": big["name"], "lr": s, "df": df, "p": stats.chi2.sf(s, df)}
        if test["p"] < 0.05:
            verdict = f"P {fmt_p(test['p'])}，小於 0.05：較複雜的模型 {big['name']} 明顯比較好，多出來的參數有必要（§5.2）。"
        else:
            verdict = f"P {fmt_p(test['p'])}，不小於 0.05：沒有證據顯示較複雜的模型 {big['name']} 比較好，用較簡單的模型 {small['name']} 即可（§5.2）。"
    else:
        warns.append({"code": "NOT_NESTED", "message": "兩個模型不是巢狀（一個不是另一個的特例），不能用概似比檢定比較，只能看 AIC（越小越好，§5.1）。"})
        verdict = None
    best_aic = min(models, key=lambda m: m["aic"])["name"]

    def desc(m):
        parts = [f"{v}（數值）" if v in m["numeric"] else f"{v}（類別）" for v in m["x"]]
        return "、".join(parts)
    print(f"【比較兩個模型：{MODEL_NAMES[kind]}】結果變數 {a.y}；n = {len(d)}（兩個模型使用同一批資料）")
    for m in models:
        print(f"- 模型 {m['name']}：{desc(m)}；參數 {m['n_params']} 個，對數概似 = {fmt(m['loglik'], 2)}，"
              f"deviance = {fmt(m['deviance'], 2)}，AIC = {fmt(m['aic'], 2)}")
    if test:
        print(f"\n概似比檢定（{test['simple']} 是 {test['complex']} 的特例）：LR = {fmt(test['lr'], 2)}，df = {test['df']}，P {fmt_p(test['p'])}")
        print(f"結論：{verdict}")
    print(f"AIC 較小的是模型 {best_aic}（AIC 兼顧配適度與參數個數，§5.1）。")
    if warns:
        print("\n注意事項")
        for w in warns:
            print(f"- [{w['code']}] {w['message']}")
    write_result({"model": "compare", "type": kind, "data": a.csv, "y": a.y, "n": len(d), "models": models,
                  "nested": nested, "lr_test": test, "verdict": verdict, "best_aic": best_aic, "warnings": warns}, out)

# ---------------------------------------------------------------- report：把分析結果整理成一份 HTML 報告

def _html(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def _img(folder, name):
    import base64
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{_html(name)}"></figure>'

def _f(x, k=3):
    if x in ("Inf", "-Inf"):
        return "∞" if x == "Inf" else "−∞"
    return "NA" if x is None else f"{x:.{k}f}"

def _p(p):
    return "NA" if p is None else ("&lt; 0.001" if p < 0.001 else f"{p:.3f}")

def _pe(p):
    return "= NA" if p is None else ("&lt; 0.001" if p < 0.001 else f"= {p:.3f}")

def _table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

def _lr_rows(r):
    return [[_html(t["variable"]), _f(t["lr"], 2), t["df"], _p(t["p"])] for t in r["lr_tests"]]

def _warn_html(r):
    if not r.get("warnings"):
        return ""
    items = "".join(f"<li><code>{_html(w['code'])}</code> {_html(w['message'])}</li>" for w in r["warnings"])
    return f'<div class="warn"><b>注意事項</b><ul>{items}</ul></div>'

def report_section(r, folder):
    m = r["model"]
    if m == "logistic":
        main = [c for c in r["coefficients"] if c["term"] != "(Intercept)"]
        sent = "".join(f"<li>其他變數固定時，{_html(c['label'])}：「{_html(r['success'])}」的勝算乘以 <b>{_f(c['or'])}</b>"
                       f"（95% CI {_f(c['or_lower'])} – {_f(c['or_upper'])}）</li>" for c in main)
        body = (f"<p>結果變數 <code>{_html(r['y'])}</code>，成功類別 = {_html(r['success'])}；n = {r['n']}（成功 {r['n_success']}）。"
                f"整體檢定 LR = {_f(r['overall']['lr'], 2)}，df = {r['overall']['df']}，P {_pe(r['overall']['p'])}。</p>"
                f"<h4>勝算比</h4><ul>{sent}</ul>"
                + _table(["項目", "β", "SE", "勝算比", f"95% CI（{_html(r['ci_type'])}）", "Wald P"],
                         [[_html(c["label"]), _f(c["estimate"]), _f(c["se"]), _f(c["or"]), f"{_f(c['or_lower'])} – {_f(c['or_upper'])}", _p(c["p"])] for c in main])
                + "<h4>每個變數的概似比檢定（拿掉該變數）</h4>" + _table(["變數", "LR", "df", "P"], _lr_rows(r))
                + f"<p>預測能力：AUC = <b>{_f(r['auc'])}</b>（0.5 = 亂猜、1 = 完美區分）</p>"
                + "<h4>預測機率（其他變數固定在平均數或最常見類別）</h4>"
                + _table(["變數", "值", "預測機率"], [[_html(e["variable"]), f"{_html(e['value'])}（{_html(e['at'])}）", _f(e["probability"])] for e in r["effects"]]))
    elif m in ("multinomial", "ordinal"):
        cats = r["categories"] if m == "multinomial" else r["levels"]
        if m == "multinomial":
            info = f"結果變數 <code>{_html(r['y'])}</code>（{len(cats)} 類，沒有順序），基準類別 = {_html(r['baseline'])}；n = {r['n']}。"
            rows = [[_html(c["equation"]), _html(c["label"]), _f(c["estimate"]), _f(c["or"]), f"{_f(c['or_lower'])} – {_f(c['or_upper'])}", _p(c["p"])]
                    for c in sorted([c for c in r["coefficients"] if c["label"] != "(Intercept)"], key=lambda c: (c["label"], c["equation"]))]
            coef = _table(["方程式", "項目", "β", "勝算比", "95% CI", "P"], rows)
        else:
            info = f"結果變數 <code>{_html(r['y'])}</code>（由低到高：{' &lt; '.join(map(_html, cats))}）；n = {r['n']}。"
            coef = ("<p>勝算比是「落在<b>較高</b>類別」的勝算倍數，對每個切點都相同。</p>"
                    + _table(["項目", "β", "較高類別的勝算比", "95% CI", "P"],
                             [[_html(c["label"]), _f(c["estimate"]), _f(c["or_higher"]), f"{_f(c['or_lower'])} – {_f(c['or_upper'])}", _p(c["p"])] for c in r["coefficients"]]))
            po = r.get("proportional_odds")
            if po:
                coef += f"<p>比例勝算假設檢定：LR = {_f(po['lr'], 2)}，df = {po['df']}，P {_pe(po['p'])}（P 小表示假設可能不成立）</p>"
        body = (f"<p>{info}整體檢定 LR = {_f(r['overall']['lr'], 2)}，df = {r['overall']['df']}，P {_pe(r['overall']['p'])}。</p>"
                + "<h4>勝算比</h4>" + coef
                + "<h4>每個變數的概似比檢定</h4>" + _table(["變數", "LR", "df", "P"], _lr_rows(r))
                + "<h4>預測機率（其他變數固定）</h4>"
                + _table(["變數", "值"] + [_html(k) for k in cats], [[_html(e["variable"]), _html(e["value"])] + [_f(e[k]) for k in cats] for e in r["effects"]]))
    elif m == "compare":
        rows = [[f"模型 {mm['name']}", _html("、".join(f"{v}（{'數值' if v in mm['numeric'] else '類別'}）" for v in mm["x"])),
                 mm["n_params"], _f(mm["loglik"], 2), _f(mm["aic"], 2)] for mm in r["models"]]
        body = (f"<p>模型：{MODEL_NAMES[r['type']]}；結果變數 <code>{_html(r['y'])}</code>；n = {r['n']}（兩個模型使用同一批資料）。</p>"
                + _table(["模型", "解釋變數", "參數個數", "對數概似", "AIC"], rows))
        if r.get("lr_test"):
            t = r["lr_test"]
            body += f"<p>概似比檢定（{t['simple']} 是 {t['complex']} 的特例）：LR = {_f(t['lr'], 2)}，df = {t['df']}，P {_pe(t['p'])}</p>"
            body += f"<p><b>結論：</b>{_html(r['verdict'])}</p>"
        body += f"<p>AIC 較小的是模型 {r['best_aic']}。</p>"
    else:
        return ""
    figs = "".join(_img(folder, p) for p in r.get("plots", []))
    title = "比較兩個模型" if m == "compare" else MODEL_NAMES[m]
    return f'<section><h2>{_html(os.path.basename(os.path.normpath(folder)))}：{title}</h2>{body}{figs}{_warn_html(r)}</section>'

REPORT_CSS = """body{font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;max-width:860px;margin:2rem auto;padding:0 16px;color:#222;line-height:1.7;background:#fff}
h1{border-bottom:3px solid #1f3a5f;padding-bottom:.3rem}h2{color:#1f3a5f;margin-top:2.5rem;border-left:5px solid #1f3a5f;padding-left:.6rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:left}th{background:#eef2f7}
figure{margin:1rem 0;text-align:center}img{max-width:100%}code{background:#f3f3f3;padding:0 .25rem;border-radius:3px}
.warn{background:#fff6e5;border-left:5px solid #e0a100;padding:.5rem 1rem;margin:1rem 0}.summary{background:#eef6ee;border-left:5px solid #2e7d32;padding:.5rem 1rem}
.foot{color:#777;font-size:.85rem;margin-top:3rem;border-top:1px solid #ddd;padding-top:.5rem}"""

def cmd_report(a):
    folders = split_arg(a.out)
    if not folders:
        sys.exit("請用 --out 指定要整理的結果資料夾（多個用逗號分隔），例如 --out output/q1,output/q2。")
    secs = []
    for f in folders:
        path = os.path.join(f, "result.json")
        if not os.path.exists(path):
            sys.exit(f"找不到 {path}；請先執行分析指令。")
        with open(path, encoding="utf-8") as fh:
            secs.append(report_section(json.load(fh), f))
    summary = ""
    if a.summary:
        with open(a.summary, encoding="utf-8") as fh:
            paras = "".join(f"<p>{_html(p.strip())}</p>" for p in fh.read().split("\n\n") if p.strip())
        summary = f'<div class="summary"><b>結論摘要</b>{paras}</div>'
    title = a.title or "類別結果模型分析報告"
    html = (f'<!doctype html><html lang="zh-TW"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_html(title)}</title><style>{REPORT_CSS}</style></head><body><h1>{_html(title)}</h1>{summary}{''.join(secs)}"
            '<p class="foot">本報告由 categorical-modeling Skill 的 run_model.py 產生；所有數字都由程式計算。'
            "方法依據：Agresti, A. (2019). <i>An Introduction to Categorical Data Analysis</i> (3rd ed.). Wiley.</p></body></html>")
    dest = a.file or "report.html"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"報告已存到：{os.path.abspath(dest)}（共 {len(secs)} 個分析）")

# ---------------------------------------------------------------- 主程式

def main():
    p = argparse.ArgumentParser(description="類別結果模型 Skill 的計算程式")
    p.add_argument("cmd", choices=["describe", "plot", "logistic", "multinomial", "ordinal", "compare", "report"])
    for opt in ("csv", "vars", "y", "x", "success", "numeric", "ref", "baseline", "levels", "out",
                "type", "x2", "numeric2", "title", "summary", "file"):
        p.add_argument(f"--{opt}", action="append")
    a = p.parse_args()
    # 同一個選項給好幾次時（例如 --ref higher=no --ref sex=F），清單型的選項全部合併，其他選項取最後一個
    for opt in ("x", "x2", "vars", "numeric", "numeric2", "ref", "out"):
        v = getattr(a, opt)
        if v is not None:
            setattr(a, opt, ",".join(s for s in v if s is not None))
    for opt in ("csv", "y", "success", "baseline", "levels", "type", "title", "summary", "file"):
        v = getattr(a, opt)
        if v is not None:
            setattr(a, opt, v[-1])
    if a.cmd == "report":
        return cmd_report(a)
    if not a.csv:
        sys.exit("請用 --csv 指定資料檔。")
    if a.cmd in ("plot", "logistic", "multinomial", "ordinal", "compare") and not a.y:
        sys.exit("請用 --y 指定結果變數。")
    if a.cmd in ("logistic", "multinomial", "ordinal", "compare") and not a.x:
        sys.exit("請用 --x 指定解釋變數（多個用逗號分隔）。")
    {"describe": cmd_describe, "plot": cmd_plot, "logistic": cmd_logistic,
     "multinomial": cmd_multinomial, "ordinal": cmd_ordinal, "compare": cmd_compare}[a.cmd](a)

if __name__ == "__main__":
    main()
