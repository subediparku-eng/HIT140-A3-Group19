import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


warnings.filterwarnings("ignore")


if len(sys.argv) > 1:
    source_file = sys.argv[1]
else:
    source_file = "match_level_104_7044.csv"

if not os.path.exists(source_file):
    raise FileNotFoundError(
        f"Cannot find {source_file}. Put the CSV next to this script "
        "or pass its path."
    )

os.makedirs("lr21_figures", exist_ok=True)

df = pd.read_csv(source_file)

target = "goal_difference"

feature_columns = [
    column
    for column in df.columns
    if column not in ("match_id", "team_a", "team_b", target)
]

if len(feature_columns) != 8:
    raise ValueError("The analysis requires exactly 8 predictors.")

X = df[feature_columns]
y = df[target]


labels = {
    "fifa_points_diff": "FIFA points diff",
    "rest_days_capped14_diff": "Rest days diff",
    "travel_km_diff": "Travel km diff",
    "cum_extra_time_min_diff": "Earlier extra-time min diff",
    "pre_wc_ppg_diff": "Pre-WC points/game diff",
    "pre_wc_clean_sheet_rate_diff": "Pre-WC clean-sheet rate diff",
    "matches_last30d_diff": "Matches in last 30 days diff",
    "host_status_diff": "Host status diff"
}


colors = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "grey": "#7f7f7f",
    "sky": "#56B4E9"
}


plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 120
})


class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, q=0.025):
        self.q = q

    def fit(self, X_data, y=None):
        X_data = np.asarray(X_data, dtype=float)
        self.lower_ = np.quantile(X_data, self.q, axis=0)
        self.upper_ = np.quantile(X_data, 1 - self.q, axis=0)
        return self

    def transform(self, X_data):
        X_data = np.asarray(X_data, dtype=float)
        return np.clip(X_data, self.lower_, self.upper_)


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def travel_transform():
    transformer = FunctionTransformer(signed_log)

    return ColumnTransformer(
        [
            ("travel", transformer, ["travel_km_diff"])
        ],
        remainder="passthrough"
    )


def select_columns(columns):
    return ColumnTransformer(
        [
            ("selected", "passthrough", columns)
        ],
        remainder="drop"
    )


ridge_alphas = np.logspace(-2, 3, 60)


def make_ridge():
    return RidgeCV(alphas=ridge_alphas)


def make_lasso():
    return LassoCV(
        alphas=np.logspace(-3, 1, 60),
        cv=5,
        max_iter=50000
    )


def make_elastic_net():
    return ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8, 0.95],
        alphas=np.logspace(-3, 1, 40),
        cv=5,
        max_iter=50000
    )


def make_model(estimator, preprocessing=None):

    steps = []

    if preprocessing is not None:
        steps.append(preprocessing)

    steps.append(StandardScaler())
    steps.append(estimator)

    return make_pipeline(*steps)


experiments = [
    (
        "Baseline",
        "Mean-only (no predictors)",
        make_pipeline(DummyRegressor())
    ),
    (
        "Linear",
        "OLS",
        make_model(LinearRegression())
    ),
    (
        "Linear",
        "Ridge",
        make_model(make_ridge())
    ),
    (
        "Linear",
        "Lasso",
        make_model(make_lasso())
    ),
    (
        "Linear",
        "ElasticNet",
        make_model(make_elastic_net())
    ),
    (
        "Linear",
        "Huber (robust)",
        make_model(HuberRegressor(max_iter=1000))
    ),
    (
        "Transform",
        "OLS + winsorise 2.5%",
        make_model(
            LinearRegression(),
            Winsorizer()
        )
    ),
    (
        "Transform",
        "OLS + signed-log travel",
        make_model(
            LinearRegression(),
            travel_transform()
        )
    ),
    (
        "Transform",
        "OLS + winsorise + signed-log travel",
        make_model(
            LinearRegression(),
            make_pipeline(
                travel_transform(),
                Winsorizer()
            )
        )
    ),
    (
        "Transform",
        "Ridge + winsorise 2.5%",
        make_model(
            make_ridge(),
            Winsorizer()
        )
    ),
    (
        "Transform",
        "Ridge + signed-log travel",
        make_model(
            make_ridge(),
            travel_transform()
        )
    ),
    (
        "Transform",
        "Ridge + winsorise + signed-log travel",
        make_model(
            make_ridge(),
            make_pipeline(
                travel_transform(),
                Winsorizer()
            )
        )
    ),
    (
        "Structure",
        "OLS no intercept (antisymmetry)",
        make_pipeline(
            StandardScaler(with_mean=False),
            LinearRegression(fit_intercept=False)
        )
    ),
    (
        "Structure",
        "Ridge no intercept (antisymmetry)",
        make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(
                alphas=ridge_alphas,
                fit_intercept=False
            )
        )
    ),
    (
        "Subset",
        "OLS: FIFA points only",
        make_model(
            LinearRegression(),
            select_columns(["fifa_points_diff"])
        )
    ),
    (
        "Subset",
        "OLS: FIFA + pre-WC points/game",
        make_model(
            LinearRegression(),
            select_columns([
                "fifa_points_diff",
                "pre_wc_ppg_diff"
            ])
        )
    ),
    (
        "Subset",
        "OLS: FIFA + PPG + clean-sheet rate",
        make_model(
            LinearRegression(),
            select_columns([
                "fifa_points_diff",
                "pre_wc_ppg_diff",
                "pre_wc_clean_sheet_rate_diff"
            ])
        )
    ),
    (
        "Benchmark",
        "Random forest (non-linear)",
        make_model(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=0
            )
        )
    ),
    (
        "Benchmark",
        "Gradient boosting (non-linear)",
        make_model(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.8,
                random_state=0
            )
        )
    )
]


n_repeats = 10
n_folds = 5

cv = RepeatedKFold(
    n_splits=n_folds,
    n_repeats=n_repeats,
    random_state=7
)

splits = list(cv.split(X, y))


def run_cross_validation(model):

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in splits:

        model_copy = clone(model)

        model_copy.fit(
            X.iloc[train_index],
            y.iloc[train_index]
        )

        predictions = model_copy.predict(
            X.iloc[test_index]
        )

        actual = y.iloc[test_index]

        rmse_scores.append(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predictions
                )
            )
        )

        mae_scores.append(
            mean_absolute_error(
                actual,
                predictions
            )
        )

        r2_scores.append(
            r2_score(
                actual,
                predictions
            )
        )

    return (
        np.array(rmse_scores),
        np.array(mae_scores),
        np.array(r2_scores)
    )


print(
    "Running",
    len(experiments),
    "experiments x",
    len(splits),
    "folds ..."
)


results = []
stored_scores = {}
ols_rmse = None


for group, name, model in experiments:

    rmse, mae, r2 = run_cross_validation(model)

    stored_scores[name] = {
        "group": group,
        "rmse": rmse
    }

    if name == "OLS":
        ols_rmse = rmse

    results.append({
        "group": group,
        "experiment": name,
        "cv_RMSE": rmse.mean(),
        "cv_RMSE_sd_over_repeats": (
            rmse.reshape(n_repeats, n_folds)
            .mean(axis=1)
            .std(ddof=1)
        ),
        "cv_MAE": mae.mean(),
        "cv_R2": r2.mean()
    })


results_table = pd.DataFrame(results)


for index, row in results_table.iterrows():

    scores = stored_scores[row["experiment"]]["rmse"]

    difference = (
        ols_rmse - scores
    ).reshape(n_repeats, n_folds)

    difference = difference.mean(axis=1)

    results_table.loc[index, "RMSE_gain_vs_OLS"] = difference.mean()

    results_table.loc[
        index,
        "better_than_OLS_in_repeats"
    ] = f"{int((difference > 0).sum())}/{n_repeats}"


results_table = (
    results_table
    .sort_values("cv_RMSE")
    .reset_index(drop=True)
)

results_table.round(4).to_csv(
    "lr21_experiment_table.csv",
    index=False
)


print(
    "\n=== EXPERIMENT TABLE "
    "(repeated 5-fold CV, 10 repeats; sorted by RMSE) ==="
)

print(
    results_table.round(3).to_string(index=False)
)


candidate_models = results_table[
    results_table["group"].isin([
        "Linear",
        "Transform",
        "Structure"
    ])
]

best_name = candidate_models.iloc[0]["experiment"]

model_lookup = {
    name: model
    for _, name, model in experiments
}

best_model = model_lookup[best_name]

print(
    "\nBest linear-regression variant by CV RMSE:",
    best_name
)


ridge_model = make_model(make_ridge())

full_ridge_rmse = stored_scores["Ridge"]["rmse"]

feature_results = []


for column in feature_columns:

    remaining_columns = [
        name
        for name in feature_columns
        if name != column
    ]

    scores = []

    for train_index, test_index in splits:

        model = clone(
            make_model(
                make_ridge(),
                select_columns(remaining_columns)
            )
        )

        model.fit(
            X.iloc[train_index],
            y.iloc[train_index]
        )

        predictions = model.predict(
            X.iloc[test_index]
        )

        score = np.sqrt(
            mean_squared_error(
                y.iloc[test_index],
                predictions
            )
        )

        scores.append(score)

    feature_results.append({
        "feature": column,
        "drop_one_RMSE_change": (
            np.mean(scores)
            - full_ridge_rmse.mean()
        )
    })


feature_analysis = pd.DataFrame(
    feature_results
).set_index("feature")


rng = np.random.default_rng(0)

permutation_scores = {
    column: []
    for column in feature_columns
}


for train_index, test_index in splits[:25]:

    model = clone(ridge_model)

    model.fit(
        X.iloc[train_index],
        y.iloc[train_index]
    )

    X_test = X.iloc[test_index].copy()

    base_prediction = model.predict(X_test)

    base_rmse = np.sqrt(
        mean_squared_error(
            y.iloc[test_index],
            base_prediction
        )
    )

    for column in feature_columns:

        changes = []

        for _ in range(20):

            shuffled = X_test.copy()

            shuffled[column] = rng.permutation(
                shuffled[column].values
            )

            prediction = model.predict(shuffled)

            shuffled_rmse = np.sqrt(
                mean_squared_error(
                    y.iloc[test_index],
                    prediction
                )
            )

            changes.append(
                shuffled_rmse - base_rmse
            )

        permutation_scores[column].append(
            np.mean(changes)
        )


feature_analysis["permutation_RMSE_increase"] = pd.Series({
    column: np.mean(values)
    for column, values in permutation_scores.items()
})


standardized_X = (
    (X - X.mean()) / X.std()
).values

design_matrix = np.column_stack([
    np.ones(len(standardized_X)),
    standardized_X
])


bootstrap_coefficients = []

for _ in range(2000):

    sample_index = rng.integers(
        0,
        len(standardized_X),
        len(standardized_X)
    )

    coefficients = np.linalg.lstsq(
        design_matrix[sample_index],
        y.values[sample_index],
        rcond=None
    )[0]

    bootstrap_coefficients.append(
        coefficients[1:]
    )


bootstrap_coefficients = np.array(
    bootstrap_coefficients
)


ols_coefficients = np.linalg.lstsq(
    design_matrix,
    y.values,
    rcond=None
)[0][1:]


feature_analysis["std_coef_OLS"] = ols_coefficients

feature_analysis["boot_CI_low"] = np.percentile(
    bootstrap_coefficients,
    2.5,
    axis=0
)

feature_analysis["boot_CI_high"] = np.percentile(
    bootstrap_coefficients,
    97.5,
    axis=0
)

feature_analysis["CI_excludes_zero"] = (
    (feature_analysis["boot_CI_low"] > 0)
    |
    (feature_analysis["boot_CI_high"] < 0)
)


feature_analysis.round(4).to_csv(
    "lr21_feature_analysis.csv"
)


print(
    "\n=== FEATURE ANALYSIS "
    "(per 1 SD of each predictor) ==="
)

print(
    feature_analysis.round(3).to_string()
)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        f"lr21_figures/{filename}.png",
        dpi=170
    )
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 4.2))

bins = np.arange(
    y.min(),
    y.max() + 2
) - 0.5

ax.hist(
    y,
    bins=bins,
    color=colors["blue"],
    edgecolor="white"
)

ax.axvline(
    y.mean(),
    color=colors["red"],
    linestyle="--",
    linewidth=2,
    label=f"Mean = {y.mean():.2f}"
)

ax.axvline(
    0,
    color="black",
    linewidth=1
)

ax.set_title(
    "Goal difference across the 104 World Cup 2026 matches"
)

ax.set_xlabel(
    "Goal difference (team A − team B)"
)

ax.set_ylabel(
    "Number of matches"
)

ax.legend(frameon=False)

save_figure(
    fig,
    "fig1_target_distribution"
)


correlation_matrix = df[
    feature_columns + [target]
].corr()

plot_labels = [
    labels.get(column, "Goal difference")
    for column in correlation_matrix.columns
]


fig, ax = plt.subplots(
    figsize=(9, 7.4)
)

image = ax.imshow(
    correlation_matrix,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)

ax.set_xticks(
    range(len(correlation_matrix))
)

ax.set_yticks(
    range(len(correlation_matrix))
)

ax.set_xticklabels(
    plot_labels,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    plot_labels
)


for i in range(len(correlation_matrix)):

    for j in range(len(correlation_matrix)):

        value = correlation_matrix.iloc[i, j]

        text_color = (
            "white"
            if abs(value) > 0.6
            else "black"
        )

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color
        )


ax.set_title(
    "Correlation between predictors and goal difference"
)

fig.colorbar(
    image,
    fraction=0.046,
    label="Pearson r"
)

ax.spines[:].set_visible(False)

save_figure(
    fig,
    "fig2_correlation_heatmap"
)


fig, ax = plt.subplots(
    figsize=(7.4, 4.8)
)

ax.scatter(
    df["fifa_points_diff"],
    y,
    s=36,
    alpha=0.75,
    color=colors["blue"],
    edgecolor="white",
    label="Match"
)


slope, intercept, correlation, p_value, _ = stats.linregress(
    df["fifa_points_diff"],
    y
)

x_values = np.linspace(
    df["fifa_points_diff"].min(),
    df["fifa_points_diff"].max(),
    50
)

ax.plot(
    x_values,
    intercept + slope * x_values,
    color=colors["red"],
    linewidth=2,
    label=(
        f"Fit: +{slope * 100:.2f} goals per "
        f"+100 pts (r = {correlation:.2f})"
    )
)

ax.axhline(
    0,
    color="black",
    linewidth=0.8
)

ax.set_title(
    "Stronger teams tend to win by more goals"
)

ax.set_xlabel(
    "FIFA points difference (team A − team B)"
)

ax.set_ylabel(
    "Goal difference"
)

ax.legend(frameon=False)

save_figure(
    fig,
    "fig3_strength_vs_goal_difference"
)


sorted_features = feature_analysis.sort_values(
    "std_coef_OLS"
)

fig, ax = plt.subplots(
    figsize=(8, 4.6)
)

positions = np.arange(
    len(sorted_features)
)

bar_colors = [
    colors["blue"]
    if significant
    else colors["grey"]
    for significant in sorted_features["CI_excludes_zero"]
]


lower_error = (
    sorted_features["std_coef_OLS"]
    - sorted_features["boot_CI_low"]
)

upper_error = (
    sorted_features["boot_CI_high"]
    - sorted_features["std_coef_OLS"]
)


ax.barh(
    positions,
    sorted_features["std_coef_OLS"],
    color=bar_colors,
    xerr=[
        lower_error,
        upper_error
    ],
    error_kw={
        "ecolor": "black",
        "capsize": 3,
        "lw": 1
    }
)

ax.set_yticks(positions)

ax.set_yticklabels([
    labels[column]
    for column in sorted_features.index
])

ax.axvline(
    0,
    color="black",
    linewidth=1
)

ax.set_title(
    "Only FIFA points clearly predicts goal difference"
)

ax.set_xlabel(
    "Change in goal difference per +1 SD of predictor "
    "(OLS, 95% bootstrap CI)"
)

save_figure(
    fig,
    "fig4_coefficients_with_CI"
)


predictions = cross_val_predict(
    clone(best_model),
    X,
    y,
    cv=KFold(
        5,
        shuffle=True,
        random_state=1
    )
)

prediction_rmse = np.sqrt(
    mean_squared_error(
        y,
        predictions
    )
)

prediction_r2 = r2_score(
    y,
    predictions
)


fig, ax = plt.subplots(
    figsize=(6, 5.6)
)

ax.scatter(
    predictions,
    y,
    s=36,
    alpha=0.7,
    color=colors["blue"],
    edgecolor="white"
)

limits = [
    min(predictions.min(), y.min()) - 0.5,
    max(predictions.max(), y.max()) + 0.5
]

ax.plot(
    limits,
    limits,
    color=colors["red"],
    linewidth=2,
    label="Perfect prediction"
)

ax.set_title(
    f"Out-of-sample predictions: {best_name}"
)

ax.set_xlabel(
    "Predicted goal difference (from held-out folds)"
)

ax.set_ylabel(
    "Actual goal difference"
)

ax.legend(
    frameon=False,
    loc="upper left"
)

ax.text(
    0.97,
    0.05,
    f"RMSE = {prediction_rmse:.2f}\n"
    f"R² = {prediction_r2:.2f}",
    transform=ax.transAxes,
    ha="right",
    bbox={
        "fc": "white",
        "ec": colors["grey"]
    }
)

save_figure(
    fig,
    "fig5_predicted_vs_actual"
)


residuals = y.values - predictions


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.4)
)

axes[0].scatter(
    predictions,
    residuals,
    s=34,
    alpha=0.7,
    color=colors["blue"],
    edgecolor="white"
)

axes[0].axhline(
    0,
    color=colors["red"],
    linewidth=2
)

axes[0].set_title(
    "Residuals vs predicted"
)

axes[0].set_xlabel(
    "Predicted goal difference"
)

axes[0].set_ylabel(
    "Residual"
)


(osm, osr), (slope, intercept, _) = stats.probplot(
    residuals,
    dist="norm"
)

axes[1].scatter(
    osm,
    osr,
    s=34,
    alpha=0.7,
    color=colors["blue"],
    edgecolor="white"
)

axes[1].plot(
    osm,
    intercept + slope * np.array(osm),
    color=colors["red"],
    linewidth=2
)

axes[1].set_title(
    f"Normal Q-Q "
    f"(Shapiro p = {stats.shapiro(residuals).pvalue:.2f})"
)

axes[1].set_xlabel(
    "Theoretical quantiles"
)

axes[1].set_ylabel(
    "Residual quantiles"
)

save_figure(
    fig,
    "fig6_residual_diagnostics"
)


comparison = (
    results_table[
        results_table["group"] != "Baseline"
    ]
    .sort_values(
        "cv_RMSE",
        ascending=False
    )
    .reset_index(drop=True)
)

baseline_rmse = results_table[
    results_table["group"] == "Baseline"
]["cv_RMSE"].iloc[0]


group_colors = {
    "Linear": colors["blue"],
    "Transform": colors["sky"],
    "Structure": colors["green"],
    "Subset": colors["orange"],
    "Benchmark": colors["grey"]
}


fig, ax = plt.subplots(
    figsize=(9.5, 7)
)


for group, group_color in group_colors.items():

    group_data = comparison[
        comparison["group"] == group
    ]

    ax.errorbar(
        group_data["cv_RMSE"],
        group_data.index,
        xerr=group_data["cv_RMSE_sd_over_repeats"],
        fmt="o",
        color=group_color,
        markersize=8,
        capsize=3,
        linewidth=1.5,
        label=group
    )


ax.axvline(
    baseline_rmse,
    color=colors["red"],
    linestyle="--",
    linewidth=2,
    label=f"Mean-only baseline ({baseline_rmse:.2f})"
)

ax.set_yticks(
    comparison.index
)

ax.set_yticklabels(
    comparison["experiment"]
)

ax.grid(
    axis="x",
    alpha=0.3
)

ax.set_title(
    "Model comparison (lower is better)"
)

ax.set_xlabel(
    "Cross-validated RMSE in goals "
    "(dots = mean; bars = spread across 10 CV repeats)"
)

ax.legend(
    frameon=False,
    loc="center right",
    bbox_to_anchor=(0.97, 0.45)
)

save_figure(
    fig,
    "fig7_model_comparison"
)


sorted_importance = feature_analysis.sort_values(
    "permutation_RMSE_increase"
)

positions = np.arange(
    len(sorted_importance)
)

fig, ax = plt.subplots(
    figsize=(8, 4.6)
)

ax.barh(
    positions - 0.2,
    sorted_importance["permutation_RMSE_increase"],
    height=0.4,
    color=colors["blue"],
    label="Permutation: RMSE rise when shuffled"
)

ax.barh(
    positions + 0.2,
    sorted_importance["drop_one_RMSE_change"],
    height=0.4,
    color=colors["orange"],
    label="Drop-one: RMSE change when removed"
)

ax.set_yticks(
    positions
)

ax.set_yticklabels([
    labels[column]
    for column in sorted_importance.index
])

ax.axvline(
    0,
    color="black",
    linewidth=1
)

ax.set_title(
    "Which predictors earn their place? (Ridge, out-of-sample)"
)

ax.set_xlabel(
    "Change in cross-validated RMSE (goals)"
)

ax.legend(
    frameon=False,
    loc="lower right"
)

save_figure(
    fig,
    "fig8_feature_importance"
)


print(
    "\nSaved lr21_experiment_table.csv, "
    "lr21_feature_analysis.csv and 8 figures "
    "in lr21_figures/"
)

