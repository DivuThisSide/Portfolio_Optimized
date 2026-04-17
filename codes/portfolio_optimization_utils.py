import numpy as np
from scipy.optimize import minimize

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    
# helper functions 

PROFILE_LAMBDA = {'conservative': 0.3, 'moderate': 0.6, 'aggressive': 1.0}

def detect_regime(rets):
    if not HMM_AVAILABLE or len(rets) < 60:
        return 'sideways'
    try:
        hmm = GaussianHMM(3, 'diag', n_iter=200, random_state=42)
        hmm.fit(rets.reshape(-1, 1))
        s    = hmm.predict(rets.reshape(-1, 1))
        rank = np.argsort(hmm.means_.flatten())
        mp   = {rank[0]: 'bear', rank[1]: 'sideways', rank[2]: 'bull'}
        return mp[s[-1]]
    except Exception:
        return 'sideways'

def regime_alpha(base, regime):
    return float(np.clip(base + {'bull':-0.02,'sideways':0.,'bear':+0.02}
                         .get(regime, 0.), 0.90, 0.99))

def port_rets(w, R):   return R @ w

def cvar_hist(pr, alpha):
    var  = np.quantile(pr, 1 - alpha)
    tail = pr[pr <= var]
    return float(-tail.mean()) if len(tail) else 0.

def cvar_param(w, mu, cov, alpha, n=5000):
    pm  = float(w @ mu)
    pv  = float(w @ cov @ w)
    std = float(np.sqrt(max(pv, 1e-12)))
    rng = np.random.default_rng(42)
    s   = rng.normal(pm, std, n)
    cut = np.quantile(s, 1 - alpha)
    t   = s[s <= cut]
    return float(-t.mean()) if len(t) else 0.

def sharpe(pr, rf, nd=252):
    ex  = pr - rf
    std = ex.std()
    return 0. if std < 1e-10 else float(ex.mean() / std * np.sqrt(nd))

def sortino(pr, rf, nd=252):
    ex = pr - rf
    dn = ex[ex < 0]
    s  = dn.std() if len(dn) > 1 else 1e-10
    return 0. if s < 1e-10 else float(ex.mean() / s * np.sqrt(nd))

def max_dd(pr):
    c = (1 + pr).cumprod()
    return float(((c - np.maximum.accumulate(c)) /
                  (np.maximum.accumulate(c) + 1e-9)).min())

def unc_bounds(sigma_hat, base_max=0.25):
    sn = (sigma_hat - sigma_hat.min()) / (sigma_hat.max() - sigma_hat.min() + 1e-9)
    return np.clip(base_max * (1. - 0.5 * sn), 0.05, base_max)

def build_obj(mu, cov, sigma, alpha, lam, ent_reg):
    def obj(w):
        v = (cvar_param(w, mu, cov, alpha, 2000)
             - lam * float(w @ mu)
             - ent_reg * float(-np.sum(w * np.log(w + 1e-9)))
             + 0.1 * float(np.sum(w * sigma)))
        return v if np.isfinite(v) else 1e6
    return obj

def ew(n): return np.ones(n) / n

def mv_max_sharpe(mu, cov):
    n = len(mu)
    def neg_s(w):
        r = float(w @ mu); v = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -r / v if v > 0 else 1e6
    r = minimize(neg_s, ew(n), method='SLSQP',
                 bounds=[(0, .4)]*n,
                 constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                 options={'maxiter':1000,'ftol':1e-9})
    w = np.clip(r.x if r.success else ew(n), 0, None)
    return w / w.sum()

print("mcvar helpers defined ")