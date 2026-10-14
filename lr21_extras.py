import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "match_level_104_7044.csv"
assert os.path.exists(SRC), f"Cannot find {SRC}. Put the CSV next to this script or pass its path."
os.makedirs("lr21_figures", exist_ok=True)

df = pd.read_csv(SRC)
TARGET = "goal_difference"
X_COLS = [c for c in df.columns if c not in ("match_id", "team_a", "team_b", TARGET)]
assert len(X_COLS) == 8
X, y = df[X_COLS], df[TARGET]
OI = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00", "grey": "#7f7f7f"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 120})


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, q=0.025): self.q = q
    def fit(self, Xa, y=None):
        Xa = np.asarray(Xa, float); self.lo_ = np.quantile(Xa, self.q, axis=0); self.hi_ = np.quantile(Xa, 1 - self.q, axis=0); return self
    def transform(self, Xa): return np.clip(np.asarray(Xa, float), self.lo_, self.hi_)

FINAL = make_pipeline(Winsorizer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 60)))   # our chosen final model
MODELS = {
    "OLS (all 8)": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge + winsorise (all 8) [final]": FINAL,
    "In-fold top-1 predictor + OLS": make_pipeline(StandardScaler(), SelectKBest(f_regression, k=1), LinearRegression()),
    "In-fold top-2 predictors + OLS": make_pipeline(StandardScaler(), SelectKBest(f_regression, k=2), LinearRegression()),
    "In-fold top-3 predictors + OLS": make_pipeline(StandardScaler(), SelectKBest(f_regression, k=3), LinearRegression()),
}
N_REP, N_FOLD = 10, 5
SPLITS = list(RepeatedKFold(n_splits=N_FOLD, n_repeats=N_REP, random_state=7).split(X, y))

fold_rmse, picks, oof = {}, {c: 0 for c in X_COLS}, {n: (np.zeros(len(y)), np.zeros(len(y))) for n in MODELS}
for name, mdl in MODELS.items():
    r = []
    for tr, te in SPLITS:
        m = clone(mdl).fit(X.iloc[tr], y.iloc[tr]); p = m.predict(X.iloc[te])
        r.append(np.sqrt(mean_squared_error(y.iloc[te], p)))
        oof[name][0][te] += p; oof[name][1][te] += 1
        if name == "In-fold top-1 predictor + OLS":
            for c in np.array(X_COLS)[m.named_steps["selectkbest"].get_support()]: picks[c] += 1
    fold_rmse[name] = np.array(r)

n_tr, n_te = len(SPLITS[0][0]), len(SPLITS[0][1])
def corrected_t(a, b):
    """Nadeau-Bengio corrected resampled t-test on per-fold RMSE differences (a - b); allows for overlapping training sets."""
    d = a - b; J = len(d); var = d.var(ddof=1) * (1 / J + n_te / n_tr)
    t = d.mean() / np.sqrt(var); return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)

rows = []
for name, r in fold_rmse.items():
    rows.append(dict(model=name, cv_RMSE=r.mean(), sd_over_folds=r.std(ddof=1)))
sel = pd.DataFrame(rows).round(4); sel.to_csv("lr21_infold_selection.csv", index=False)
print("=== IN-FOLD SELECTION: selection happens inside each training fold, so there is no peeking ===")
print(sel.to_string(index=False))
print("\nPredictor chosen as the single best in each of", len(SPLITS), "training folds:", {k: v for k, v in picks.items() if v})

pairs = [("Ridge + winsorise (all 8) [final]", "OLS (all 8)"),
         ("In-fold top-1 predictor + OLS", "OLS (all 8)"),
         ("In-fold top-1 predictor + OLS", "Ridge + winsorise (all 8) [final]"),
         ("In-fold top-2 predictors + OLS", "Ridge + winsorise (all 8) [final]")]
prow = []
for a, b in pairs:
    md, t, p = corrected_t(fold_rmse[a], fold_rmse[b])
    prow.append(dict(model_A=a, model_B=b, mean_RMSE_A_minus_B=md, t=t, p_value=p, A_significantly_better=(p < .05 and md < 0)))
pt = pd.DataFrame(prow).round(4); pt.to_csv("lr21_paired_tests.csv", index=False)
print("\n=== PAIRED TESTS (corrected resampled t-test; negative difference = model A has lower error) ===")
print(pt.to_string(index=False))


pred = oof["Ridge + winsorise (all 8) [final]"][0] / oof["Ridge + winsorise (all 8) [final]"][1]
out = df[["team_a", "team_b"]].copy(); out["actual_GD"] = y.values; out["predicted_GD"] = pred.round(2); out["error"] = (y.values - pred).round(2)
out["abs_error"] = out.error.abs(); miss = out.sort_values("abs_error", ascending=False).head(10).reset_index(drop=True)
miss.drop(columns="abs_error").to_csv("lr21_biggest_misses.csv", index=False)
print("\n=== 10 BIGGEST MISSES (actual vs out-of-sample predicted goal difference, team A minus team B) ===")
print(miss.drop(columns="abs_error").to_string(index=False))
print(f"\nMean |error| = {out.abs_error.mean():.2f} goals; share of matches within 1 goal: {(out.abs_error <= 1).mean():.0%}; within 2 goals: {(out.abs_error <= 2).mean():.0%}")

mask = (np.abs(pred) >= .5) & (y.values != 0)
print(f"Winner called correctly in {np.mean(np.sign(pred[mask]) == np.sign(y.values[mask])):.0%} of {mask.sum()} decisive matches where |prediction| >= 0.5")

m = miss.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5.6))
for i, r in m.iterrows(): ax.plot([r.predicted_GD, r.actual_GD], [i, i], color=OI["grey"], lw=2, zorder=1)
ax.scatter(m.predicted_GD, m.index, s=90, color=OI["blue"], zorder=2, label="Model prediction (out-of-sample)")
ax.scatter(m.actual_GD, m.index, s=90, color=OI["red"], zorder=3, label="Actual result")
ax.set_yticks(m.index); ax.set_yticklabels([f"{a} v {b}" for a, b in zip(m.team_a, m.team_b)]); ax.axvline(0, color="k", lw=1)
ax.set(title="The 10 matches the model got most wrong", xlabel="Goal difference (first-named team \u2212 second-named team)"); ax.grid(axis="x", alpha=.3)
ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig("lr21_figures/fig9_biggest_misses.png", dpi=170); plt.close(fig)
print("\nSaved lr21_infold_selection.csv, lr21_paired_tests.csv, lr21_biggest_misses.csv, lr21_figures/fig9_biggest_misses.png")
