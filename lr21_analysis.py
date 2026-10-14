import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys, os
SRC = sys.argv[1] if len(sys.argv) > 1 else "match_level_104_7044.csv"  
assert os.path.exists(SRC), f"Cannot find {SRC}. Put the CSV next to the script or pass its path: python lr21_analysis.py path/to/file.csv"

def ols_fit(Xdf, yv):
    """Plain-numpy OLS with classical inference (no statsmodels needed)."""
    A = np.column_stack([np.ones(len(Xdf)), Xdf.values]); n, k = A.shape
    XtXi = np.linalg.inv(A.T @ A); b = XtXi @ A.T @ yv.values
    fit = A @ b; r = yv.values - fit; sse = r @ r; sst = ((yv - yv.mean())**2).sum()
    s2 = sse / (n - k); se = np.sqrt(np.diag(XtXi) * s2); t = b / se; p = 2 * stats.t.sf(abs(t), n - k)
    r2 = 1 - sse / sst; adj = 1 - (1 - r2) * (n - 1) / (n - k)
    F = ((sst - sse) / (k - 1)) / s2; fp = stats.f.sf(F, k - 1, n - k)
    h = np.einsum("ij,jk,ik->i", A, XtXi, A); cook = (r**2 / (k * s2)) * h / (1 - h)**2
    ci = np.column_stack([b - stats.t.ppf(.975, n-k) * se, b + stats.t.ppf(.975, n-k) * se])
    tab = pd.DataFrame({"coef": b, "std_err": se, "t": t, "p": p, "ci_low": ci[:, 0], "ci_high": ci[:, 1]}, index=["const"] + list(Xdf.columns))
    return dict(tab=tab, fit=fit, resid=r, r2=r2, adj=adj, fp=fp, cook=cook, A=A)

def vif(Xdf):
    out = {}
    for c in Xdf.columns:
        o = ols_fit(Xdf.drop(columns=c), Xdf[c]); out[c] = 1 / (1 - o["r2"])
    return pd.Series(out)

def breusch_pagan(A, r):
    g = r**2 / (r @ r / len(r)); beta = np.linalg.lstsq(A, g, rcond=None)[0]
    ess = ((A @ beta - g.mean())**2).sum(); lm = ess / 2
    return stats.chi2.sf(lm, A.shape[1] - 1)

def durbin_watson(r): return np.sum(np.diff(r)**2) / np.sum(r**2)

df = pd.read_csv(SRC)
TARGET = "goal_difference"
X_COLS = [c for c in df.columns if c not in ("match_id", "team_a", "team_b", TARGET)]
assert len(X_COLS) == 8
X, y = df[X_COLS], df[TARGET]

print("SUMMARY "); print(df[X_COLS + [TARGET]].describe().T.round(3))
print("\nMissing:", int(df.isna().sum().sum()), "| duplicate match_id:", int(df.match_id.duplicated().sum()))
print("\n CORRELATION WITH TARGET")
print(df[X_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3))
print("\n=== VIF ===")
print(vif(X).round(2))


z = (X - X.mean()) / X.std()
print("\nRows with |z|>3:\n", z[(z.abs() > 3).any(axis=1)].round(2).join(df[["team_a", "team_b"]]).iloc[:, -2:])

ols = ols_fit(X, y); res = ols["resid"]; cook = ols["cook"]
print("\n=== OLS (all 104 rows, for inference/diagnostics) ===")
print(ols["tab"].round(4)); print(f"R2={ols['r2']:.3f} adjR2={ols['adj']:.3f} F p={ols['fp']:.4f}")
print("Shapiro-Wilk p:", round(stats.shapiro(res).pvalue, 4),
      "| Breusch-Pagan p:", round(breusch_pagan(ols["A"], res), 4),
      "| Durbin-Watson:", round(durbin_watson(res), 3))
print("Max Cook's D:", round(cook.max(), 3), "| rows > 4/n:", int((cook > 4/len(df)).sum()))
ols["tab"].round(4).to_csv("lr21_ols_coefficients.csv")


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 60))),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 60), cv=5, max_iter=50000)),
}
rows = []
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=1)
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    s = cross_validate(m, X, y, cv=cv, scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
    rows.append(dict(model=name,
        holdout_RMSE=np.sqrt(mean_squared_error(yte, p)), holdout_MAE=mean_absolute_error(yte, p), holdout_R2=r2_score(yte, p),
        cv_RMSE=-s["test_neg_root_mean_squared_error"].mean(), cv_MAE=-s["test_neg_mean_absolute_error"].mean(), cv_R2=s["test_r2"].mean()))
base = np.sqrt(((yte - ytr.mean())**2).mean())
tab = pd.DataFrame(rows).round(3); print("\n VALIDATION RESULTS "); print(tab.to_string(index=False))
print("Baseline (predict train mean) hold-out RMSE:", round(base, 3))
for name in ("Ridge", "Lasso"):
    est = models[name][-1]; print(f"{name} chosen alpha:", round(float(est.alpha_), 4))
std_coef = pd.DataFrame({n: models[n][-1].coef_ for n in models}, index=X_COLS)
std_coef["OLS_std"] = models["OLS"][-1].coef_
print("\nStandardised coefficients (per 1 SD):"); print(std_coef[["OLS_std"]].join(std_coef[["Ridge", "Lasso"]]).round(3))
tab.to_csv("lr21_results_table.csv", index=False)

# ---- plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].hist(y, bins=range(-5, 8), edgecolor="k"); ax[0,0].set_title("Goal difference")
cm = df[X_COLS + [TARGET]].corr()
im = ax[0,1].imshow(cm, cmap="coolwarm", vmin=-1, vmax=1); ax[0,1].set_title("Correlation matrix")
ax[0,1].set_xticks(range(len(cm))); ax[0,1].set_xticklabels(range(len(cm))); ax[0,1].set_yticks(range(len(cm)))
ax[0,1].set_yticklabels([f"{i}:{c[:14]}" for i, c in enumerate(cm.columns)], fontsize=6); fig.colorbar(im, ax=ax[0,1], fraction=.046)
ax[0,2].scatter(df.fifa_points_diff, y, alpha=.7); ax[0,2].set_xlabel("FIFA points diff"); ax[0,2].set_ylabel("Goal diff"); ax[0,2].set_title("Strength vs goal difference")
ax[1,0].scatter(ols["fit"], res, alpha=.7); ax[1,0].axhline(0, color="r"); ax[1,0].set_title("Residuals vs fitted")
stats.probplot(res, dist="norm", plot=ax[1,1]); ax[1,1].set_title("Residual Q-Q")
ax[1,2].stem(cook); ax[1,2].set_title("Cook's distance")
plt.tight_layout(); plt.savefig("lr21_diagnostics.png", dpi=130)
print("\nSaved lr21_results_table.csv, lr21_diagnostics.png")
