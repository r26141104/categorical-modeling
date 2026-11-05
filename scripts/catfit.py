"""catfit.py — 類別結果模型的配適函數（只用 numpy 與 Python 標準函式庫）

不需要 statsmodels、scipy，也不需要網路；claude.ai 網頁版的程式執行環境可以直接使用。

依據：Agresti, A. (2019). An Introduction to Categorical Data Analysis (3rd ed.)
  二元 logistic（§4.1–4.6、§5.1–5.3）：Newton-Raphson（= Fisher scoring = IRLS，canonical link）
  baseline-category logit（§6.1）：Newton-Raphson
  累積 logit（§6.2–6.3）：Newton-Raphson，觀察資訊矩陣；參數化 logit P(Y ≤ j) = θ_j − x'β
  完全分離與無限估計值（§5.3）：最大概似估計不存在時，概似函數變平，但參數一直往無限大移動；
    偵測到時 status = "separation"，sep 標出估計值為無限大的參數。
"""

import math
from statistics import NormalDist

import numpy as np

_N01 = NormalDist()
_EPS = np.finfo(float).eps

# ================================================================ 分布函數與數值工具

def norm_sf(z):
    """標準常態的右尾機率 P(Z > z)。"""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))

def norm_ppf(q):
    """標準常態的分位數。"""
    return _N01.inv_cdf(float(q))

def _gamma_p_series(a, x):
    s = term = 1.0 / a
    ap = a
    for _ in range(10000):
        ap += 1.0
        term *= x / ap
        s += term
        if abs(term) < abs(s) * 1e-17:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))

def _gamma_q_cf(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-17:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

def gammaincc(a, x):
    """正規化的上不完全 gamma 函數 Q(a, x)。"""
    if x <= 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_cf(a, x)

def chi2_sf(x, df):
    """卡方分布的右尾機率 P(χ²_df > x)。"""
    x = float(x)
    if math.isnan(x):
        return float("nan")
    if x <= 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)

def chi2_ppf(q, df):
    """卡方分布的分位數；df = 1 時用 z²，其他 df 用二分法。"""
    if df == 1:
        return norm_ppf((1.0 + q) / 2.0) ** 2
    lo, hi = 0.0, max(1.0, df)
    while chi2_sf(hi, df) > 1 - q:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, df) > 1 - q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def brentq(f, a, b, xtol=2e-12, rtol=4 * _EPS, maxiter=200):
    """Brent 法求根（f(a)、f(b) 必須異號）。"""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) 與 f(b) 同號")
    if fa == 0:
        return a
    if fb == 0:
        return b
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * rtol * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2.0 * m * s, 1.0 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    raise RuntimeError("Brent 法沒有收斂")

def rankdata(v):
    """等級（同分取平均等級），和 scipy.stats.rankdata 的預設相同。"""
    v = np.asarray(v, dtype=float)
    o = np.argsort(v, kind="mergesort")
    s = v[o]
    r = np.empty(len(v))
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        r[o[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return r

def expit(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _solve(A, g):
    try:
        return np.linalg.solve(A, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, g, rcond=None)[0]

def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)

class Fit:
    """配適結果。status：ok（收斂）／separation（完全或準完全分離，§5.3）／not_converged（沒有收斂）。"""
    def __init__(self, **kw):
        self.__dict__.update(kw)

def _newton(par, loglik_grad_info, maxiter, feasible=None):
    """共用的 Newton-Raphson：每一步解 I·δ = g，對數概似沒有增加就把步長減半。
    回傳 (參數, 對數概似, 狀態, 最後一步, 疊代次數)。
      ok       ：步長小於 1e-10（真正收斂）
      flat     ：對數概似已經不變，但參數還在大幅移動 → 最大概似估計不存在（分離）
      maxiter  ：疊代次數用完
      stuck    ：步長減半到極小仍無法增加對數概似（數值問題，視為沒有收斂）"""
    ll, g, info = loglik_grad_info(par)
    last = np.zeros_like(par)
    for it in range(1, maxiter + 1):
        step = _solve(info, g)
        t = 1.0
        while True:
            new = par + t * step
            if feasible is None or feasible(new):
                ll_new, g_new, info_new = loglik_grad_info(new)
                if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * (abs(ll) + 1.0):
                    break
            t /= 2.0
            if t < 1e-12:
                return par, ll, "stuck", last, it
        dll = ll_new - ll
        last = new - par
        par, ll, g, info = new, ll_new, g_new, info_new
        if np.max(np.abs(last)) < 1e-10:
            return par, ll, "ok", last, it
        if it >= 5 and abs(dll) < 1e-13 * (abs(ll) + 0.1) and np.max(np.abs(last)) > 1e-4:
            return par, ll, "flat", last, it
    return par, ll, "maxiter", last, maxiter

def _status(raw, extreme, last):
    if raw == "ok":
        return "ok", np.zeros(len(last), dtype=bool)
    if extreme and raw != "stuck":
        big = np.abs(last)
        return "separation", big > max(1e-6, 1e-3 * big.max())
    return "not_converged", np.ones(len(last), dtype=bool)

# ================================================================ 二元 logistic 迴歸（§4.1–4.6）

def logit_loglik(y, eta):
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

def logit_fit(y, X, offset=None, maxiter=100):
    """logit π = Xβ (+ offset)。X：n × p 的 numpy 陣列（含截距欄）。"""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)

    def lgi(b):
        eta = X @ b + off
        mu = expit(eta)
        w = mu * expit(-eta)                         # π(1 − π)，不會因相減而損失精度
        return logit_loglik(y, eta), X.T @ (y - mu), X.T @ (X * w[:, None])

    # 起始值和 R 的 glm 相同：μ₀ = (y + 0.5) / 2，先做一次加權最小平方（有 offset 時也穩定）
    mu0 = (y + 0.5) / 2.0
    w0 = mu0 * (1 - mu0)
    z0 = np.log(mu0 / (1 - mu0)) - off + (y - mu0) / w0
    b0 = _solve(X.T @ (X * w0[:, None]), X.T @ (w0 * z0))
    b, ll, raw, last, it = _newton(b0, lgi, maxiter)
    eta = X @ b + off
    mu = expit(eta)
    w = mu * expit(-eta)
    cov = _inv(X.T @ (X * w[:, None]))
    extreme = bool(np.any((mu < 1e-8) | (mu > 1 - 1e-8)))
    status, sep = _status(raw, extreme, last)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return Fit(params=b, cov=cov, se=se, llf=ll, deviance=-2.0 * ll, fitted=mu, status=status, sep=sep, iterations=it)

def logit_predict(b, Xn):
    return expit(np.asarray(Xn, dtype=float) @ b)

# ================================================================ baseline-category logit（§6.1）

def _softmax_base(eta):
    """eta：n × K（K = J − 1）；回傳 n × J 的機率，第 0 欄是基準類別。"""
    m = np.maximum(0.0, eta.max(axis=1, keepdims=True))
    e0 = np.exp(-m)
    ek = np.exp(eta - m)
    s = e0 + ek.sum(axis=1, keepdims=True)
    return np.column_stack([e0 / s, ek / s])

def mnlogit_fit(yc, X, J, maxiter=100):
    """log(π_k / π_0) = x'β_k，k = 1, …, J − 1；yc 是 0, …, J − 1 的代碼，0 是基準類別。
    參數向量依方程式排列：[β_1 (p 個), β_2, …]；params 是 p × (J − 1) 的矩陣。"""
    yc = np.asarray(yc, dtype=int)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    K = J - 1
    Y = np.zeros((n, J))
    Y[np.arange(n), yc] = 1.0

    def lgi(par):
        B = par.reshape(K, p).T
        P = _softmax_base(X @ B)
        ll = float(np.sum(np.log(np.clip(P[np.arange(n), yc], 1e-300, None))))
        g = (X.T @ (Y[:, 1:] - P[:, 1:])).T.reshape(-1)
        info = np.empty((K * p, K * p))
        for k in range(K):
            for l in range(K):
                w = P[:, k + 1] * ((k == l) - P[:, l + 1])
                info[k * p:(k + 1) * p, l * p:(l + 1) * p] = X.T @ (X * w[:, None])
        return ll, g, info

    par, ll, raw, last, it = _newton(np.zeros(K * p), lgi, maxiter)
    _, _, info = lgi(par)
    cov = _inv(info)
    P = _softmax_base(X @ par.reshape(K, p).T)
    status, sep = _status(raw, bool(np.any(P < 1e-8)), last)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return Fit(params=par.reshape(K, p).T, se=se.reshape(K, p).T, cov=cov, llf=ll,
               sep=sep.reshape(K, p).T, status=status, iterations=it)

def mnlogit_predict(B, Xn):
    return _softmax_base(np.asarray(Xn, dtype=float) @ B)

# ================================================================ 累積 logit（§6.2–6.3）

def _pdiff(tu, tl):
    """σ(tu) − σ(tl)，兩者都接近 1 時改算 σ(−tl) − σ(−tu)，避免相減損失精度。"""
    hi = (tu + tl) > 0
    a = expit(tu) - expit(tl)
    b = expit(-tl) - expit(-tu)
    return np.where(hi, b, a)

def cumlogit_fit(yc, X, J, parallel=True, start=None, maxiter=200):
    """logit P(Y ≤ j) = θ_j − x'β（parallel = True，比例勝算）或 θ_j − x'β_j（parallel = False）。
    yc：0, …, J − 1（由低到高）。參數向量：[θ_1, …, θ_{J−1}, β]；非比例時 β 依切點排列。"""
    yc = np.asarray(yc, dtype=int)
    X = np.asarray(X, dtype=float).reshape(len(yc), -1)
    n, p = X.shape
    K = J - 1
    q = K + (p if parallel else K * p)
    up = np.where(yc < K, yc, -1)                    # 上界切點（最高類別沒有）
    lo = np.where(yc >= 1, yc - 1, -1)               # 下界切點（最低類別沒有）

    def zmat(cut):
        Z = np.zeros((n, q))
        ok = cut >= 0
        Z[np.where(ok)[0], cut[ok]] = 1.0
        if p:
            if parallel:
                Z[ok, K:] = -X[ok]
            else:
                for i in np.where(ok)[0]:
                    Z[i, K + cut[i] * p:K + (cut[i] + 1) * p] = -X[i]
        return Z, ok

    Zu, oku = zmat(up)
    Zl, okl = zmat(lo)

    def parts(par):
        tu = Zu @ par
        tl = Zl @ par
        Fu = np.where(oku, expit(tu), 1.0)
        Fl = np.where(okl, expit(tl), 0.0)
        pi = np.where(oku & okl, _pdiff(tu, tl), Fu - Fl)
        return tu, tl, Fu, Fl, pi

    def feasible(par):
        return bool(np.all(parts(par)[4] > 0))

    def lgi(par):
        tu, tl, Fu, Fl, pi = parts(par)
        if np.any(pi <= 0):
            return -np.inf, None, None
        fu = np.where(oku, Fu * (1 - Fu), 0.0)
        fl = np.where(okl, Fl * (1 - Fl), 0.0)
        fpu = fu * (1 - 2 * Fu)
        fpl = fl * (1 - 2 * Fl)
        S = (fu[:, None] * Zu - fl[:, None] * Zl) / pi[:, None]
        g = S.sum(axis=0)
        H = Zu.T @ (Zu * (fpu / pi)[:, None]) - Zl.T @ (Zl * (fpl / pi)[:, None]) - S.T @ S
        return float(np.sum(np.log(pi))), g, -H

    if start is None:
        cnt = np.bincount(yc, minlength=J).astype(float)
        cum = np.clip(np.cumsum(cnt)[:-1] / n, 1e-4, 1 - 1e-4)
        par0 = np.concatenate([np.log(cum / (1 - cum)), np.zeros(q - K)])
    else:
        th, be = start
        be = np.asarray(be, dtype=float)
        par0 = np.concatenate([th, be if parallel else np.tile(be, K)])
    par, ll, raw, last, it = _newton(par0, lgi, maxiter, feasible)
    _, _, info = lgi(par)
    cov = _inv(info)
    tu, tl, Fu, Fl, pi = parts(par)
    lim = math.log((1 - 1e-8) / 1e-8)               # |t| 超過這個值，累積機率在 1e-8 以內接近 0 或 1
    extreme = bool(np.any(oku & (np.abs(tu) > lim)) or np.any(okl & (np.abs(tl) > lim)))
    status, sep = _status(raw, extreme, last)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return Fit(theta=par[:K], beta=par[K:], se_theta=se[:K], se_beta=se[K:], cov=cov, llf=ll,
               status=status, sep_beta=sep[K:], sep_theta=sep[:K], iterations=it)

def cumlogit_predict(theta, beta, Xn):
    """比例勝算模型的各類別預測機率（n × J）。"""
    Xn = np.asarray(Xn, dtype=float).reshape(-1, len(beta))
    t = theta[None, :] - (Xn @ beta)[:, None]
    cum = np.column_stack([np.zeros(len(Xn)), expit(t), np.ones(len(Xn))])
    return np.diff(cum, axis=1)

def null_loglik(yc):
    nn = np.bincount(np.asarray(yc, dtype=int))
    nn = nn[nn > 0]
    return float(np.sum(nn * np.log(nn / nn.sum())))
