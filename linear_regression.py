import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)   
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# checking the data
df = pd.read_csv("team_match_208.csv")
print(df.head())
print(df.info())
print ((df["cross_border_travel"]).value_counts())
print ((df["match_id"]).nunique())
print ((df["team_id"]).nunique())
print ((df["knockout_stage"]).value_counts())
print (((df["team_id"])==(df["opponent_id"])).sum())


# Descriptive statistics
predictor_vars = ["fifa_points_gap",
                    "cross_border_travel",
                    "team_rest_days_capped14",
                    "team_travel_km",
                    "local_kickoff_hour",
                    "knockout_stage",
                    "prior_matches_same_venue",
                    "team_cum_extra_time_min"
                   
                    ]
analysis_vars = predictor_vars + ["team_goals"]
analysis =df[analysis_vars].describe().to_csv("descriptive_stats.csv")
print(analysis)



# Plot the target-variable distribution
plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="team_goals",
    color="steelblue"
)

# Display frequency above each bar
for container in ax.containers:
    ax.bar_label(container)

plt.title("Distribution of Team Goals")
plt.xlabel("Goals scored by a team")
plt.ylabel("Number of team-match observations")
plt.tight_layout()

plt.savefig(
    "figures/ team_goals_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# visualizing the explanatory variables
# Numerical explanatory variables
numerical_predictors = [
    "fifa_points_gap",
    "team_rest_days_capped14",
    "team_travel_km",
    "team_cum_extra_time_min",
    "prior_matches_same_venue",
    "local_kickoff_hour"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for variable, axis in zip(numerical_predictors, axes):
    sns.histplot(
        data=df,
        x=variable,
        bins=15,
        color="steelblue",
        ax=axis
    )

    axis.set_title(variable.replace("_", " ").title())
    axis.set_xlabel("")
    axis.set_ylabel("Frequency")

fig.suptitle(
    "Distributions of Numerical Explanatory Variables",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "figures/numerical_predictor_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Binary explanatory variables
binary_predictors = [
    "cross_border_travel",
    "knockout_stage"
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for variable, axis in zip(binary_predictors, axes):
    sns.countplot(
        data=df,
        x=variable,
        color="steelblue",
        ax=axis
    )

    for container in axis.containers:
        axis.bar_label(container)

    axis.set_title(variable.replace("_", " ").title())
    axis.set_xlabel("0 = No, 1 = Yes")
    axis.set_ylabel("Number of observations")

plt.tight_layout()

plt.savefig(
    "figures/binary_predictor_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# data analysis
# Calculate the Pearson correlation matrix
correlation_matrix = df[analysis_vars].corr(method="pearson")

print("\nCorrelation matrix:")
print(correlation_matrix.round(3))

# Save the correlation values
correlation_matrix.round(3).to_csv(
    "results/correlation_matrix.csv"
)

# Create the correlation heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Correlation Matrix of Model Variables")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "figures/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Multicollinearity analysis using VIF
X_vif = df[predictor_vars].astype(float)
X_vif_with_constant = sm.add_constant(X_vif)

vif_results = pd.DataFrame({
    "Variable": predictor_vars,
    "VIF": [
        variance_inflation_factor(
            X_vif_with_constant.values,
            column_index
        )
        for column_index in range(
            1,
            X_vif_with_constant.shape[1]
        )
    ]
})

vif_results = vif_results.sort_values(
    by="VIF",
    ascending=False
)

print("\nVariance Inflation Factors:")
print(vif_results.round(3))

vif_results.round(3).to_csv(
    "results/vif_results.csv",
    index=False
)

# Define predictors, target and match groups and split the data into training and testing sets
X = df[predictor_vars]
y = df["team_goals"]
groups = df["match_id"]

print("\nModel components:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of match groups:", groups.nunique())

# Create a grouped training-validation split
group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, validation_indices = next(
    group_split.split(
        X,
        y,
        groups=groups
    )
)

# Create training data
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
groups_train = groups.iloc[train_indices]

# Create validation data
X_validation = X.iloc[validation_indices]
y_validation = y.iloc[validation_indices]
groups_validation = groups.iloc[validation_indices]

overlapping_matches = set(groups_train).intersection(
    set(groups_validation)
)

print("\nGrouped split:")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_validation))
print("Training matches:", groups_train.nunique())
print("Validation matches:", groups_validation.nunique())
print("Overlapping matches:", len(overlapping_matches))

# creating a baseline model
# Calculate regression performance metrics
def evaluate_model(model_name, actual, predicted):
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    r_squared = r2_score(actual, predicted)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r_squared
    }

# Baseline prediction: training-set mean
baseline_value = y_train.mean()

baseline_predictions = np.full(
    len(y_validation),
    baseline_value
)

baseline_results = evaluate_model(
    "Baseline Mean",
    y_validation,
    baseline_predictions
)

print("\nBaseline prediction:")
print(round(baseline_value, 3))

print("\nBaseline validation results:")
print(pd.DataFrame([baseline_results]).round(3))

# Building and training the multiple linear regression model
linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)
linear_train_predictions = linear_model.predict(X_train)
linear_validation_predictions = linear_model.predict(X_validation)

linear_train_results = evaluate_model(
    "Linear Regression - Training",
    y_train,
    linear_train_predictions
)

linear_results = evaluate_model(
    "Linear Regression",
    y_validation,
    linear_validation_predictions
)

print("\nLinear Regression training results:")
print(pd.DataFrame([linear_train_results]).round(3))

print("\nLinear Regression validation results:")
print(pd.DataFrame([linear_results]).round(3))

# creating actual vs predicted plot
prediction_results = df.iloc[validation_indices][
    ["match_id", "team", "opponent", "team_goals"]
].reset_index(drop=True)

prediction_results["predicted_team_goals"] = (
    linear_validation_predictions
)

prediction_results["residual"] = (
    prediction_results["team_goals"]
    - prediction_results["predicted_team_goals"]
)

print("\nActual versus predicted goals:")
print(prediction_results.head(10).round(3))

prediction_results.round(3).to_csv(
    "results/linear_regression_predictions.csv",
    index=False
)

# Fitting  an OLS model on the training data for statistical interpretation
X_train_with_constant = sm.add_constant(X_train)

ols_model = sm.OLS(
    y_train,
    X_train_with_constant
).fit()

print("\nOLS Regression Summary:")
print(ols_model.summary())

with open(
    "results/ols_summary.txt",
    "w",
    encoding="utf-8"
) as summary_file:
    summary_file.write(
        ols_model.summary().as_text()
    )

# Check influential observations using Cook's distance
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]

# Common guideline for identifying influential observations
limit = 4 / len(X_train)

print("\nCook's distance limit:", round(limit, 4))
print("Number above limit:", (cooks_d > limit).sum())

# Add Cook's distance to the training observations
training_rows = df.iloc[train_indices].copy()
training_rows["cooks_distance"] = cooks_d

most_influential = training_rows[
    [
        "match_id",
        "team",
        "opponent",
        "team_goals",
        "cooks_distance"
    ]
].sort_values(
    "cooks_distance",
    ascending=False
)

print("\nMost influential observations:")
print(most_influential.head(10).round(3))

most_influential.to_csv(
    "results/influential_observations.csv",
    index=False
)

# Plot Cook's distance
plt.figure(figsize=(10, 5))
plt.scatter(range(len(cooks_d)), cooks_d)

plt.axhline(
    limit,
    color="red",
    linestyle="--",
    label="4/n guideline"
)

plt.title("Cook's Distance")
plt.xlabel("Training observation")
plt.ylabel("Cook's distance")
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/cooks_distance.png",
    dpi=300
)

plt.close()

# Ridge regression
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# Alpha values to test
ridge_parameters = {
    "ridge__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100,
        1000
    ]
}

# Keep both teams from each match in the same fold
group_cv = GroupKFold(n_splits=5)

ridge_search = GridSearchCV(
    estimator=ridge_model,
    param_grid=ridge_parameters,
    scoring="neg_root_mean_squared_error",
    cv=group_cv
)

# Tune Ridge using only the training data
ridge_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\nBest Ridge alpha:")
print(ridge_search.best_params_["ridge__alpha"])

# Use the best Ridge model
best_ridge_model = ridge_search.best_estimator_

ridge_train_predictions = best_ridge_model.predict(X_train)
ridge_validation_predictions = best_ridge_model.predict(
    X_validation
)

ridge_train_results = evaluate_model(
    "Ridge Regression - Training",
    y_train,
    ridge_train_predictions
)

ridge_results = evaluate_model(
    "Ridge Regression",
    y_validation,
    ridge_validation_predictions
)

print("\nRidge training results:")
print(pd.DataFrame([ridge_train_results]).round(3))

print("\nRidge validation results:")
print(pd.DataFrame([ridge_results]).round(3))

# final results comparison
model_comparison = pd.DataFrame([
    baseline_results,
    linear_results,
    ridge_results
])

model_comparison = model_comparison.sort_values(
    "RMSE"
)

print("\nFinal model comparison:")
print(model_comparison.round(3))

model_comparison.round(3).to_csv(
    "results/model_comparison.csv",
    index=False
)

final_predictions = df.iloc[validation_indices][
    ["match_id", "team", "opponent", "team_goals"]
].reset_index(drop=True)

final_predictions["predicted_team_goals"] = (
    ridge_validation_predictions
)

final_predictions["residual"] = (
    final_predictions["team_goals"]
    - final_predictions["predicted_team_goals"]
)

final_predictions.round(3).to_csv(
    "results/ridge_predictions.csv",
    index=False
)
# Linear Regression validation residuals
linear_validation_residuals = (
    y_validation.to_numpy()
    - linear_validation_predictions
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual versus predicted goals
axes[0].scatter(
    y_validation,
    linear_validation_predictions,
    color="steelblue",
    alpha=0.7
)

lowest_value = min(
    y_validation.min(),
    linear_validation_predictions.min()
)

highest_value = max(
    y_validation.max(),
    linear_validation_predictions.max()
)

axes[0].plot(
    [lowest_value, highest_value],
    [lowest_value, highest_value],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)

axes[0].set_title("Actual Versus Predicted Goals")
axes[0].set_xlabel("Actual team goals")
axes[0].set_ylabel("Predicted team goals")
axes[0].legend()


# Validation residuals
axes[1].scatter(
    linear_validation_predictions,
    linear_validation_residuals,
    color="steelblue",
    alpha=0.7
)

axes[1].axhline(
    y=0,
    color="red",
    linestyle="--"
)

axes[1].set_title("Validation Residuals")
axes[1].set_xlabel("Predicted team goals")
axes[1].set_ylabel("Residual")


plt.tight_layout()

plt.savefig(
    "figures/linear_regression_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# pllotting the ridge regression validation residuals
# Ridge validation residuals
ridge_residuals = (
    y_validation.to_numpy()
    - ridge_validation_predictions
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual versus predicted goals
axes[0].scatter(
    y_validation,
    ridge_validation_predictions,
    color="darkgreen",
    alpha=0.7
)

lowest_value = min(
    y_validation.min(),
    ridge_validation_predictions.min()
)

highest_value = max(
    y_validation.max(),
    ridge_validation_predictions.max()
)

axes[0].plot(
    [lowest_value, highest_value],
    [lowest_value, highest_value],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)

axes[0].set_title("Ridge: Actual Versus Predicted Goals")
axes[0].set_xlabel("Actual team goals")
axes[0].set_ylabel("Predicted team goals")
axes[0].legend()


# Residual plot
axes[1].scatter(
    ridge_validation_predictions,
    ridge_residuals,
    color="darkgreen",
    alpha=0.7
)

axes[1].axhline(
    y=0,
    color="red",
    linestyle="--"
)

axes[1].set_title("Ridge Validation Residuals")
axes[1].set_xlabel("Predicted team goals")
axes[1].set_ylabel("Residual")

plt.tight_layout()

plt.savefig(
    "figures/ridge_regression_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Lasso regression
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=100000))
])

# Test a range of regularisation strengths
lasso_parameters = {
    "lasso__alpha": np.logspace(
        -4,
        1,
        40
    )
}

lasso_search = GridSearchCV(
    estimator=lasso_model,
    param_grid=lasso_parameters,
    scoring="neg_root_mean_squared_error",
    cv=group_cv
)

# Tune Lasso using only the training data
lasso_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

best_lasso_model = lasso_search.best_estimator_
best_lasso_alpha = lasso_search.best_params_["lasso__alpha"]

print("\nBest Lasso alpha:")
print(round(best_lasso_alpha, 6))

# Generate predictions
lasso_train_predictions = best_lasso_model.predict(X_train)

lasso_validation_predictions = best_lasso_model.predict(
    X_validation
)

# Evaluate predictions
lasso_train_results = evaluate_model(
    "Lasso Regression - Training",
    y_train,
    lasso_train_predictions
)

lasso_results = evaluate_model(
    "Lasso Regression",
    y_validation,
    lasso_validation_predictions
)

print("\nLasso training results:")
print(pd.DataFrame([lasso_train_results]).round(3))

print("\nLasso validation results:")
print(pd.DataFrame([lasso_results]).round(3))

lasso_coefficients = pd.DataFrame({
    "Variable": predictor_vars,
    "Standardised coefficient":
        best_lasso_model.named_steps["lasso"].coef_
})

lasso_coefficients["Selected"] = (
    lasso_coefficients["Standardised coefficient"] != 0
)

print("\nLasso coefficients:")
print(lasso_coefficients.round(4))

lasso_coefficients.round(4).to_csv(
    "results/lasso_coefficients.csv",
    index=False
)


model_comparison = pd.DataFrame([
    baseline_results,
    linear_results,
    ridge_results,
    lasso_results
])

model_comparison = model_comparison.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\nFinal model comparison:")
print(model_comparison.round(3))

model_comparison.round(3).to_csv(
    "results/model_comparison.csv",
    index=False
)

# Lasso validation residuals
lasso_residuals = (
    y_validation.to_numpy()
    - lasso_validation_predictions
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual versus predicted
axes[0].scatter(
    y_validation,
    lasso_validation_predictions,
    color="darkorange",
    alpha=0.7
)

lowest_value = min(
    y_validation.min(),
    lasso_validation_predictions.min()
)

highest_value = max(
    y_validation.max(),
    lasso_validation_predictions.max()
)

axes[0].plot(
    [lowest_value, highest_value],
    [lowest_value, highest_value],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)

axes[0].set_title("Lasso: Actual Versus Predicted Goals")
axes[0].set_xlabel("Actual team goals")
axes[0].set_ylabel("Predicted team goals")
axes[0].legend()


# Residuals
axes[1].scatter(
    lasso_validation_predictions,
    lasso_residuals,
    color="darkorange",
    alpha=0.7
)

axes[1].axhline(
    y=0,
    color="red",
    linestyle="--"
)

axes[1].set_title("Lasso Validation Residuals")
axes[1].set_xlabel("Predicted team goals")
axes[1].set_ylabel("Residual")

plt.tight_layout()

plt.savefig(
    "figures/lasso_regression_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()