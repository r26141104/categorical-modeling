"""比對測試：新的 run_model.py（只用 numpy、pandas）和 statsmodels、R 的結果，每個數字差距要小於 1e-6。

只在本機執行，Skill 執行時不需要這個檔案。
- Skill 一律透過 tests/run_without_statsmodels.py 執行（擋掉 statsmodels、scipy），證明執行時不依賴它們。
- 參考值：statsmodels（收斂標準 1e-14 以下）與 R（tests/reference_R.json，由 tests/make_reference_R.R 產生）。
- 差距的算法：|新 − 參考| / max(1, |參考|)；數字小於 1 時就是絕對差距。

執行（在 repo 根目錄）：
    Rscript tests/make_reference_R.R        # R 的參考值（已附在 repo，可略過）
    python tests/test_against_reference.py   # 需要 numpy、pandas、scipy、statsmodels、scikit-learn
結果會印在畫面上，並寫到 tests/RESULTS.md；有任何一項 ≥ 1e-6 就以錯誤結束。
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "tests", "run_without_statsmodels.py")
DATA = os.path.join(REPO, "data")
POL = os.path.join(REPO, "report", "skill-test", "tidy-check", "polviews_tidy.csv")
TOL = 1e-6
R = json.load(open(os.path.join(REPO, "tests", "reference_R.json"), encoding="utf-8"))
TMP = tempfile.mkdtemp(prefix="catfit-test-")
warnings.filterwarnings("ignore")
ROWS, INFO = [], []

# ---------------------------------------------------------------- 工具

def run_skill(name, *args, no_mpl=False):
    out = os.path.join(TMP, name)
    cmd = [sys.executable, "-I", RUN] + (["--no-matplotlib"] if no_mpl else []) + list(args) + ["--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name} 執行失敗：\n{p.stdout}\n{p.stderr}")
    with open(os.path.join(out, "result.json"), encoding="utf-8") as fh:
        return json.load(fh), p.stdout

def num(v):
    if v == "Inf":
        return math.inf
    if v == "-Inf":
        return -math.inf
    return float("nan") if v is None else float(v)

def check(case, what, got, ref, source):
    g = np.atleast_1d(np.asarray([num(v) for v in np.atleast_1d(got)], dtype=float))
    r = np.atleast_1d(np.asarray(ref, dtype=float))
    assert g.shape == r.shape, (case, what, g.shape, r.shape)
    diff = float(np.max(np.abs(g - r) / np.maximum(1.0, np.abs(r)))) if len(r) else 0.0
    ROWS.append({"case": case, "what": what, "n": len(r), "diff": diff, "source": source})

def as_str(v):
    return str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)

def design(d, spec, const=True):
    """spec：[(變數, "num") 或 (變數, [參考組, 其他類別…])]；欄名和 R 相同（變數名 + 類別）。和 Skill 的程式分開寫。"""
    cols = {"(Intercept)": np.ones(len(d))} if const else {}
    for v, kind in spec:
        if kind == "num":
            cols[v] = d[v].astype(float).values
        else:
            s = d[v].map(as_str)
            for lv in kind[1:]:
                cols[f"{v}{lv}"] = (s == lv).astype(float).values
    return pd.DataFrame(cols)

def newdata(d, spec):
    """Skill 的預測機率規則：其他變數固定在平均數（數值）或最常見類別；要比較的變數用 Q1、Q3（相同時用最小、最大值）或每個類別。"""
    typ = {v: (float(d[v].astype(float).mean()) if k == "num" else d[v].map(as_str).value_counts().idxmax()) for v, k in spec}
    rows = []
    for v, k in spec:
        if k == "num":
            s = d[v].astype(float)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            vals, names = ([s.min(), s.max()], ["最小值", "最大值"]) if q1 == q3 else ([q1, q3], ["Q1", "Q3"])
        else:
            vals, names = k, k
        for val, nm in zip(vals, names):
            t = dict(typ)
            t[v] = val
            rows.append((v, nm, t))
    nd = pd.DataFrame([t for _, _, t in rows])
    return [(v, nm) for v, nm, _ in rows], nd

def sm_logit(y, X, offset=None):
    return sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit(tol=1e-15, maxiter=300)

def profile_ref(y, X):
    """profile likelihood 信賴區間的參考值：statsmodels 配適 + scipy 的 brentq，精確解出 2[ℓ(β̂) − ℓ_p(b)] = χ²₁,0.95。"""
    fit = sm_logit(y, X)
    crit = stats.chi2.ppf(0.95, 1)
    out = []
    for col in X.columns:
        b, se = fit.params[col], fit.bse[col]
        others = X.drop(columns=col)
        g = lambda val: 2 * (fit.llf - sm_logit(y, others, offset=X[col].values * val).llf) - crit
        bounds = []
        for sgn in (-1, 1):
            k = 1.0
            while g(b + sgn * k * se) < 0:
                k *= 1.6
            lo, hi = sorted([b + sgn * k / 1.6 * se, b + sgn * k * se])
            bounds.append(optimize.brentq(g, lo, hi, xtol=1e-14, rtol=1e-15))
        out.append(bounds)
    return np.array(out)

def r_by_name(ref, names, rename=lambda s: s):
    L = lambda v: v if isinstance(v, list) else [v]          # jsonlite 會把只有一個值的向量存成純量
    m = {rename(n): (e, s) for n, e, s in zip(L(ref["names"]), L(ref["estimate"]), L(ref["se"]))}
    return np.array([m[n][0] for n in names]), np.array([m[n][1] for n in names])

# ---------------------------------------------------------------- 二元 logistic

def logistic_case(case, d, yname, success, spec, args, r_key, r_rename=lambda s: s):
    res, _ = run_skill(case, "logistic", *args)
    y = (d[yname].map(as_str) == success).astype(float).values
    X = design(d, spec)
    f = sm_logit(y, X)
    terms = [c["term"] for c in res["coefficients"]]
    assert terms == list(X.columns), (terms, list(X.columns))
    est = [c["estimate"] for c in res["coefficients"]]
    check(case, "係數 β", est, f.params.values, "statsmodels")
    check(case, "標準誤", [c["se"] for c in res["coefficients"]], f.bse.values, "statsmodels")
    check(case, "Wald z", [c["z"] for c in res["coefficients"]], f.tvalues.values, "statsmodels")
    check(case, "Wald P", [c["p"] for c in res["coefficients"]], f.pvalues.values, "statsmodels")
    check(case, "勝算比", [c["or"] for c in res["coefficients"]], np.exp(f.params.values), "statsmodels")
    ci = profile_ref(y, X)
    assert res["ci_type"] == "profile likelihood"
    check(case, "profile CI 下界（log 勝算比）", [math.log(c["or_lower"]) for c in res["coefficients"]], ci[:, 0], "statsmodels + scipy brentq")
    check(case, "profile CI 上界（log 勝算比）", [math.log(c["or_upper"]) for c in res["coefficients"]], ci[:, 1], "statsmodels + scipy brentq")
    rb, rs = r_by_name(R[r_key], terms, r_rename)
    check(case, "係數 β", est, rb, "R glm")
    check(case, "標準誤", [c["se"] for c in res["coefficients"]], rs, "R glm")
    # 概似比檢定
    f0 = sm_logit(y, X[["(Intercept)"]])
    ref_lr, ref_p = [], []
    for v, k in spec:
        fr = sm_logit(y, design(d, [s for s in spec if s[0] != v]))
        ref_lr.append(fr.deviance - f.deviance)
        ref_p.append(stats.chi2.sf(ref_lr[-1], 1 if k == "num" else len(k) - 1))
    check(case, "各變數 LR", [t["lr"] for t in res["lr_tests"]], ref_lr, "statsmodels")
    check(case, "各變數 LR 的 P", [t["p"] for t in res["lr_tests"]], ref_p, "statsmodels + scipy")
    check(case, "各變數 LR", [t["lr"] for t in res["lr_tests"]], list(R[r_key]["lr"].values()), "R glm")
    check(case, "整體 LR", [res["overall"]["lr"]], [f0.deviance - f.deviance], "statsmodels")
    check(case, "整體 LR", [res["overall"]["lr"]], [R[r_key]["overall_lr"]], "R glm")
    check(case, "整體 LR 的 P", [res["overall"]["p"]], [stats.chi2.sf(f0.deviance - f.deviance, X.shape[1] - 1)], "scipy")
    check(case, "AUC", [res["auc"]], [roc_auc_score(y, np.round(f.fittedvalues, 12))], "scikit-learn")
    check(case, "AUC", [res["auc"]], [R[r_key]["auc"]], "R pROC")
    keys, nd = newdata(d, spec)
    pr = f.predict(design(nd, spec))
    got = {(e["variable"], e["value"]): e["probability"] for e in res["effects"]}
    check(case, "預測機率", [got[k] for k in keys], np.asarray(pr), "statsmodels")
    return res, f

# ---------------------------------------------------------------- baseline-category logit

def multinomial_case(case, d, yname, base, spec, args, r_key):
    res, _ = run_skill(case, "multinomial", *args)
    lv = res["categories"]
    order = [base] + [k for k in lv if k != base]
    yc = d[yname].map(as_str).map({k: i for i, k in enumerate(order)}).values
    X = design(d, spec)
    f = sm.MNLogit(yc, X).fit(method="newton", tol=1e-14, maxiter=300, disp=False)
    est, se, names = [], [], []
    for j, k in enumerate(order[1:]):
        eq = [o for o in lv if o != base].index(k) + 1
        for t in X.columns:
            names.append(f"{t}:{eq}")
            est.append(f.params.iloc[X.columns.get_loc(t), j])
            se.append(f.bse.iloc[X.columns.get_loc(t), j])
    got = {c["term"]: c for c in res["coefficients"]}
    g = lambda key: [got[n][key] for n in names]
    check(case, "係數 β", g("estimate"), est, "statsmodels")
    check(case, "標準誤", g("se"), se, "statsmodels")
    check(case, "Wald z", g("z"), np.array(est) / np.array(se), "statsmodels")
    check(case, "Wald P", g("p"), 2 * stats.norm.sf(np.abs(np.array(est) / np.array(se))), "scipy")
    check(case, "Wald CI 下界（log 勝算比）", [math.log(v) for v in g("or_lower")], np.array(est) - 1.96 * np.array(se), "statsmodels")
    check(case, "Wald CI 上界（log 勝算比）", [math.log(v) for v in g("or_upper")], np.array(est) + 1.96 * np.array(se), "statsmodels")
    rb, rs = r_by_name(R[r_key], names)
    check(case, "係數 β", g("estimate"), rb, "R VGAM")
    check(case, "標準誤", g("se"), rs, "R VGAM")
    f0 = sm.MNLogit(yc, X[["(Intercept)"]]).fit(method="newton", tol=1e-14, maxiter=300, disp=False)
    check(case, "整體 LR", [res["overall"]["lr"]], [2 * (f.llf - f0.llf)], "statsmodels")
    check(case, "整體 LR", [res["overall"]["lr"]], [R[r_key]["overall_lr"]], "R VGAM")
    ref_lr = [2 * (f.llf - sm.MNLogit(yc, design(d, [s for s in spec if s[0] != v])).fit(method="newton", tol=1e-14, maxiter=300, disp=False).llf) for v, _ in spec]
    check(case, "各變數 LR", [t["lr"] for t in res["lr_tests"]], ref_lr, "statsmodels")
    check(case, "各變數 LR", [t["lr"] for t in res["lr_tests"]], list(R[r_key]["lr"].values()), "R VGAM")
    check(case, "各變數 LR 的 P", [t["p"] for t in res["lr_tests"]], [stats.chi2.sf(s, t["df"]) for s, t in zip(ref_lr, res["lr_tests"])], "scipy")
    keys, nd = newdata(d, spec)
    pr = np.asarray(f.predict(design(nd, spec)))
    eff = {(e["variable"], e["value"]): e for e in res["effects"]}
    check(case, "預測機率", [eff[k][c] for k in keys for c in lv], [pr[i, order.index(c)] for i in range(len(keys)) for c in lv], "statsmodels")
    return res

# ---------------------------------------------------------------- 累積 logit

def ordinal_case(case, d, yname, levels, spec, args, r_key):
    res, _ = run_skill(case, "ordinal", *args)
    yc = d[yname].map(as_str).map({k: i for i, k in enumerate(levels)}).values.astype(int)
    X = design(d, spec, const=False)
    terms = [c["term"] for c in res["coefficients"]]
    rb, rs = r_by_name(R[r_key], terms)
    est = [c["estimate"] for c in res["coefficients"]]
    check(case, "係數 β", est, rb, "R ordinal::clm")
    check(case, "標準誤", [c["se"] for c in res["coefficients"]], rs, "R ordinal::clm")
    check(case, "Wald z", [c["z"] for c in res["coefficients"]], rb / rs, "R ordinal::clm")
    check(case, "Wald P", [c["p"] for c in res["coefficients"]], 2 * stats.norm.sf(np.abs(rb / rs)), "R + scipy")
    check(case, "較高類別的勝算比", [c["or_higher"] for c in res["coefficients"]], np.exp(rb), "R ordinal::clm")
    check(case, "Wald CI 下界（log 勝算比）", [math.log(c["or_lower"]) for c in res["coefficients"]], rb - 1.96 * rs, "R ordinal::clm")
    check(case, "Wald CI 上界（log 勝算比）", [math.log(c["or_upper"]) for c in res["coefficients"]], rb + 1.96 * rs, "R ordinal::clm")
    check(case, "整體 LR", [res["overall"]["lr"]], [R[r_key]["overall_lr"]], "R ordinal::clm")
    check(case, "各變數 LR", [t["lr"] for t in res["lr_tests"]], list(R[r_key]["lr"].values()), "R ordinal::clm")
    check(case, "各變數 LR 的 P", [t["p"] for t in res["lr_tests"]], [stats.chi2.sf(s, t["df"]) for s, t in zip(R[r_key]["lr"].values(), res["lr_tests"])], "scipy")
    check(case, "比例勝算檢定 LR", [res["proportional_odds"]["lr"]], [R[r_key]["po_lr"]], "R ordinal::clm（nominal）")
    check(case, "比例勝算檢定 P", [res["proportional_odds"]["p"]], [stats.chi2.sf(R[r_key]["po_lr"], res["proportional_odds"]["df"])], "scipy")
    # 預測機率：用 R 的 θ、β 算
    keys, nd = newdata(d, spec)
    Xn = design(nd, spec, const=False).values
    th = np.array(R[r_key]["theta"])
    cum = 1 / (1 + np.exp(-(th[None, :] - (Xn @ rb)[:, None])))
    pr = np.diff(np.column_stack([np.zeros(len(Xn)), cum, np.ones(len(Xn))]), axis=1)
    eff = {(e["variable"], e["value"]): e for e in res["effects"]}
    check(case, "預測機率", [eff[k][c] for k in keys for c in levels], pr.reshape(-1), "R ordinal::clm")
    # statsmodels 的 OrderedModel（BFGS，數值微分的標準誤）只當參考，列在說明裡
    om = OrderedModel(yc, X, distr="logit").fit(method="bfgs", gtol=1e-12, maxiter=20000, disp=False)
    INFO.append(f"{case}：statsmodels OrderedModel 和 R clm 的差距：β {np.max(np.abs(np.asarray(om.params)[:X.shape[1]] - rb)):.1e}，"
                f"標準誤 {np.max(np.abs(np.asarray(om.bse)[:X.shape[1]] - rs)):.1e}（OrderedModel 用數值微分，所以拿 clm 當主要參考值）")
    return res

# ---------------------------------------------------------------- 分離與分布函數

def separation_cases():
    tmp = os.path.join(TMP, "sep")
    os.makedirs(tmp, exist_ok=True)
    # 完全分離：x = 1–3 都失敗、4–6 都成功（§5.3）
    pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": ["no", "no", "no", "yes", "yes", "yes"]}).to_csv(f"{tmp}/c.csv", index=False)
    res, out = run_skill("sep_complete", "logistic", "--csv", f"{tmp}/c.csv", "--y", "y", "--x", "x", "--numeric", "x", "--success", "yes")
    c = {k["term"]: k for k in res["coefficients"]}
    ok = (any(w["code"] == "SEPARATION" for w in res["warnings"]) and c["x"]["estimate"] == "Inf" and c["x"]["se"] is None
          and c["x"]["p"] is None and c["x"]["or_lower"] is None)
    ROWS.append({"case": "完全分離（logistic）", "what": "警告 SEPARATION、x 的估計值輸出 Inf、SE／P／CI 輸出 NA", "n": 1, "diff": 0.0 if ok else 1.0, "source": "§5.3"})
    check("完全分離（logistic）", "整體 LR（極限值 = 虛無 deviance = 12 ln 2）", [res["overall"]["lr"]], [12 * math.log(2)], "精確值")
    # 準完全分離：x = 4 有一個失敗、一個成功，其他完全分開
    pd.DataFrame({"x": [1, 2, 3, 4, 4, 5, 6, 7], "y": ["no", "no", "no", "no", "yes", "yes", "yes", "yes"]}).to_csv(f"{tmp}/q.csv", index=False)
    res, _ = run_skill("sep_quasi", "logistic", "--csv", f"{tmp}/q.csv", "--y", "y", "--x", "x", "--numeric", "x", "--success", "yes")
    c = {k["term"]: k for k in res["coefficients"]}
    ok = any(w["code"] == "SEPARATION" for w in res["warnings"]) and c["x"]["estimate"] == "Inf" and c["x"]["se"] is None
    ROWS.append({"case": "準完全分離（logistic）", "what": "警告 SEPARATION、x 的估計值輸出 Inf", "n": 1, "diff": 0.0 if ok else 1.0, "source": "§5.3"})
    check("準完全分離（logistic）", "整體 LR（極限值 = 16 ln 2 − 4 ln 2）", [res["overall"]["lr"]], [12 * math.log(2)], "精確值")
    # baseline-category logit：b 組從未出現類別 A
    pd.DataFrame({"g": list("aaaaabbbbb"), "y": list("AABCCBBCCC")}).to_csv(f"{tmp}/m.csv", index=False)
    res, _ = run_skill("sep_multi", "multinomial", "--csv", f"{tmp}/m.csv", "--y", "y", "--x", "g")
    c = {k["term"]: k for k in res["coefficients"]}
    ok = (any(w["code"] == "SEPARATION" for w in res["warnings"]) and c["gb:1"]["estimate"] == "-Inf" and c["gb:1"]["se"] is None
          and isinstance(c["gb:2"]["estimate"], float) and abs(c["gb:2"]["estimate"]) < 10)
    ROWS.append({"case": "分離（baseline-category logit）", "what": "只有 gb:1（A vs C）輸出 −Inf，其他係數正常", "n": 1, "diff": 0.0 if ok else 1.0, "source": "§5.3"})
    # 累積 logit：a 組都在 1–2、b 組都在 3
    pd.DataFrame({"g": list("aaabbb"), "y": [1, 1, 2, 3, 3, 3]}).to_csv(f"{tmp}/o.csv", index=False)
    res, _ = run_skill("sep_ord", "ordinal", "--csv", f"{tmp}/o.csv", "--y", "y", "--x", "g", "--levels", "1,2,3")
    c = {k["term"]: k for k in res["coefficients"]}
    ok = any(w["code"] == "SEPARATION" for w in res["warnings"]) and c["gb"]["estimate"] == "Inf" and c["gb"]["se"] is None
    ROWS.append({"case": "分離（累積 logit）", "what": "警告 SEPARATION、gb 的估計值輸出 Inf", "n": 1, "diff": 0.0 if ok else 1.0, "source": "§5.3"})

def distribution_cases():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import catfit as cf
    xs = [1e-4, 0.01, 0.1, 0.5, 1, 2, 3.84, 5, 10, 17.2, 30, 59.4, 100, 300]
    got, ref = [], []
    for df in range(1, 31):
        for x in xs:
            got.append(cf.chi2_sf(x, df))
            ref.append(stats.chi2.sf(x, df))
    check("分布函數", "卡方右尾機率（df 1–30，420 個點）", got, ref, "scipy")
    rel = max(abs(g - r) / r for g, r in zip(got, ref) if r > 1e-300)
    INFO.append(f"卡方右尾機率的最大「相對」誤差 {rel:.1e}（連 1e-100 這麼小的 P 值也一樣準）")
    zs = np.linspace(-8, 8, 161)
    check("分布函數", "常態右尾機率（−8 到 8）", [cf.norm_sf(z) for z in zs], stats.norm.sf(zs), "scipy")
    check("分布函數", "卡方分位數（0.95，df 1–10）", [cf.chi2_ppf(0.95, k) for k in range(1, 11)], [stats.chi2.ppf(0.95, k) for k in range(1, 11)], "scipy")
    rng = np.random.default_rng(1)
    v = rng.integers(0, 20, 500).astype(float)
    check("分布函數", "等級（含同分，AUC 用）", cf.rankdata(v), stats.rankdata(v), "scipy")

# ---------------------------------------------------------------- 主程式

def main():
    st = pd.read_csv(os.path.join(DATA, "student_por.csv"))
    crabs = pd.read_csv(os.path.join(DATA, "crabs.csv"))
    al = pd.read_csv(os.path.join(DATA, "alligators.csv"))
    pol = pd.read_csv(POL, encoding="utf-8-sig")
    S, C, A = os.path.join(DATA, "student_por.csv"), os.path.join(DATA, "crabs.csv"), os.path.join(DATA, "alligators.csv")

    # 課本例題
    logistic_case("馬蹄蟹 y ~ width（§4.1）", crabs, "y", "1", [("width", "num")],
                  ["--csv", C, "--y", "y", "--x", "width", "--numeric", "width"], "crabs_width")
    logistic_case("馬蹄蟹 y ~ width + color（§4.4）", crabs, "y", "1", [("width", "num"), ("color", ["1", "2", "3", "4"])],
                  ["--csv", C, "--y", "y", "--x", "width,color", "--numeric", "width"], "crabs_color")
    multinomial_case("鱷魚 food ~ length（§6.1）", al, "food", "O", [("length", "num")],
                     ["--csv", A, "--y", "food", "--x", "length", "--numeric", "length", "--baseline", "O"], "alligators")
    ordinal_case("政治意識形態 ~ party + gender（§6.2）", pol, "ideology", ["y1", "y2", "y3", "y4", "y5"],
                 [("party", ["dem", "repub"]), ("gender", ["female", "male"])],
                 ["--csv", POL, "--y", "ideology", "--x", "party,gender", "--levels", "y1,y2,y3,y4,y5"], "polviews")
    # 示範資料（實測的三題與模型比較）
    q1spec = [("studytime", "num"), ("failures", "num"), ("higher", ["no", "yes"]), ("sex", ["F", "M"])]
    q1args = ["--csv", S, "--y", "pass", "--x", "studytime,failures,higher,sex", "--success", "pass"]
    r_q1, f_q1 = logistic_case("高中生 pass（實測第 1 題）", st, "pass", "pass", q1spec, q1args + ["--numeric", "studytime,failures"], "student_q1")
    q1cat = [("studytime", "num"), ("failures", ["0", "1", "2", "3"]), ("higher", ["no", "yes"]), ("sex", ["F", "M"])]
    r_cat, f_cat = logistic_case("高中生 pass，failures 當類別", st, "pass", "pass", q1cat, q1args + ["--numeric", "studytime"], "student_q1cat",
                                 lambda s: s.replace("failuresf", "failures"))
    ordinal_case("高中生 Walc ~ sex（實測第 2 題）", st, "Walc", ["1", "2", "3", "4", "5"], [("sex", ["F", "M"])],
                 ["--csv", S, "--y", "Walc", "--x", "sex", "--levels", "1,2,3,4,5"], "student_q2")
    multinomial_case("高中生 reason ~ address + school（實測第 3 題）", st, "reason", "other",
                     [("address", ["R", "U"]), ("school", ["GP", "MS"])],
                     ["--csv", S, "--y", "reason", "--x", "address,school", "--baseline", "other"], "student_q3")
    # compare：對數概似、deviance、AIC、LR 檢定
    res, _ = run_skill("compare_logistic", "compare", "--type", "logistic", *q1args, "--numeric", "studytime,failures", "--numeric2", "studytime")
    m = {mm["name"]: mm for mm in res["models"]}
    check("模型比較（logistic）", "對數概似", [m["A"]["loglik"], m["B"]["loglik"]], [f_q1.llf, f_cat.llf], "statsmodels")
    check("模型比較（logistic）", "deviance", [m["A"]["deviance"], m["B"]["deviance"]], [f_q1.deviance, f_cat.deviance], "statsmodels")
    check("模型比較（logistic）", "AIC", [m["A"]["aic"], m["B"]["aic"]], [f_q1.aic, f_cat.aic], "statsmodels")
    check("模型比較（logistic）", "AIC", [m["A"]["aic"], m["B"]["aic"]],
          [-2 * R["student_q1"]["loglik"] + 2 * 5, -2 * R["student_q1cat"]["loglik"] + 2 * 7], "R glm")
    check("模型比較（logistic）", "LR 檢定與 P", [res["lr_test"]["lr"], res["lr_test"]["p"]],
          [2 * (f_cat.llf - f_q1.llf), stats.chi2.sf(2 * (f_cat.llf - f_q1.llf), 2)], "statsmodels + scipy")
    res, _ = run_skill("compare_ordinal", "compare", "--type", "ordinal", "--csv", S, "--y", "Walc", "--x", "sex", "--x2", "sex,address", "--levels", "1,2,3,4,5")
    m = {mm["name"]: mm for mm in res["models"]}
    lA, lB = R["student_q2"]["loglik"], R["student_q2_addr"]["loglik"]
    check("模型比較（累積 logit）", "對數概似與 AIC", [m["A"]["loglik"], m["B"]["loglik"], m["A"]["aic"], m["B"]["aic"]],
          [lA, lB, -2 * lA + 2 * 5, -2 * lB + 2 * 6], "R ordinal::clm")
    check("模型比較（累積 logit）", "LR 檢定", [res["lr_test"]["lr"]], [2 * (lB - lA)], "R ordinal::clm")
    # 沒有 matplotlib 也能算，數字完全相同
    res2, _ = run_skill("no_matplotlib", "logistic", *q1args, "--numeric", "studytime,failures", no_mpl=True)
    check("沒有 matplotlib", "所有係數（和有 matplotlib 時相比）", [c["estimate"] for c in res2["coefficients"]],
          [c["estimate"] for c in r_q1["coefficients"]], "同一程式")
    ROWS.append({"case": "沒有 matplotlib", "what": "照常輸出 result.json、不出圖", "n": 1, "diff": 0.0 if res2["plots"] == [] else 1.0, "source": "—"})
    separation_cases()
    distribution_cases()
    # check 指令（claude.ai 上要先跑的自我檢查）
    p = subprocess.run([sys.executable, "-I", RUN, "check"], capture_output=True, text=True)
    ROWS.append({"case": "check 指令", "what": "擋掉 statsmodels、scipy 時「全部通過」", "n": 1,
                 "diff": 0.0 if "全部通過" in p.stdout else 1.0, "source": "—"})
    report()

def old_vs_new():
    """舊版（statsmodels）實測時存下的 result.json，和新版重跑的結果比較（只是說明，不列入 1e-6 的判定）。"""
    lines = []
    T = os.path.join(REPO, "report", "skill-test")
    pairs = [("round1/q1", "高中生 pass（實測第 1 題）"), ("round1/q2", "高中生 Walc ~ sex（實測第 2 題）"),
             ("round1/q3", "高中生 reason ~ address + school（實測第 3 題）"), ("round2/q1_cat", "高中生 pass，failures 當類別"),
             ("round2/q1_compare", "compare_logistic")]
    def flat(x, p=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from flat(v, f"{p}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from flat(v, f"{p}[{i}]")
        else:
            yield p, x
    for old, new in pairs:
        a = dict(flat(json.load(open(os.path.join(T, old, "result.json"), encoding="utf-8"))))
        b = dict(flat(json.load(open(os.path.join(TMP, new, "result.json"), encoding="utf-8"))))
        diffs = [(abs(a[k] - b[k]) / max(1, abs(a[k])), k) for k in a if k in b and isinstance(a[k], float) and isinstance(b[k], float)]
        worst = max(diffs)
        lines.append(f"| `{old}` | {len(diffs)} | {worst[0]:.1e} | `{worst[1]}` |")
    return lines

def report():
    bad = [r for r in ROWS if not r["diff"] < TOL]
    old = old_vs_new()
    md = ["# 新程式（只用 numpy、pandas）和 statsmodels／R 的比對結果", "",
          "由 `tests/test_against_reference.py` 產生。Skill 一律在**擋掉 statsmodels、scipy、scikit-learn** 的情況下執行。",
          "差距 = |新 − 參考| / max(1, |參考|)，每一項都要小於 1e-6。", "",
          f"**結果：{len(ROWS) - len(bad)} / {len(ROWS)} 項通過**（共比對 {sum(r['n'] for r in ROWS)} 個數字）", "",
          "| 例子 | 項目 | 數字個數 | 最大差距 | 參考 |", "|---|---|---|---|---|"]
    md += [f"| {r['case']} | {r['what']} | {r['n']} | {r['diff']:.1e} | {r['source']} |" for r in ROWS]
    md += ["", "## 說明", ""] + [f"- {s}" for s in INFO]
    md += ["", "## 和舊版（statsmodels）實測結果的比較", "",
           "舊版實測時存下的 `result.json`，和新版重跑的結果比較。舊版的收斂標準較寬（statsmodels GLM 預設 1e-8、OrderedModel 用 BFGS 與數值微分），",
           "所以差距在 1e-6 到 1e-4 之間；新版和收斂到 1e-14 的 statsmodels、R 的差距都小於 1e-6（上表），新版比舊版更準。", "",
           "| 舊版結果 | 數字個數 | 最大差距 | 差距最大的項目 |", "|---|---|---|---|"] + old
    with open(os.path.join(REPO, "tests", "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    if bad:
        print("\n不通過：", bad)
        sys.exit(1)

if __name__ == "__main__":
    main()
