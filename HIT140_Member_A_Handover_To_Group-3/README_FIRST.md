# HIT140 Assessment 3 Group Handover

Member A has completed the data collection, source documentation, data preparation and validation work for Objective 2. Send this complete package to all group members so everyone uses the same dataset version.

## Start here

- Member B uses `data/processed/match_level_104.csv`.
- Member C uses `data/processed/team_match_208.csv`.
- Member D uses `docs/member_A_report_text.md`, `docs/member_A_data_dictionary.xlsx` and the validation files when preparing the final report.
- Everyone should read the variable definitions before starting analysis.
- Do not rename, remove or recalculate columns without informing the whole group.

## What Member A completed

- Downloaded official FIFA World Cup match, ranking and team-history data.
- Documented FBref as the second source for manual fixture and score checking.
- Produced the 104-row match-level dataset for Linear Regression 2.1.
- Produced the 208-row team-match dataset for Linear Regression 2.2.
- Created exactly eight explanatory variables for each regression task.
- Limited the shared predictor concepts to FIFA strength, rest, travel and cumulative extra time.
- Prepared the source manifest, field dictionary, supporting lookup tables and Python scripts.
- Confirmed 104 unique matches, 208 team rows, 48 teams and no missing model values.
- Reconciled team goals and goal differences with the official match registry.

## Member B task

Use `match_level_104.csv` to predict `goal_difference`.

1. Exclude `match_id`, `team_a`, `team_b` and the target from the explanatory-variable matrix.
2. Complete descriptive statistics, data visualisations and correlation analysis.
3. Check unusual values, multicollinearity and linear regression assumptions.
4. Build and evaluate a multiple linear regression model.
5. Compare it with at least one suitable alternative, such as Ridge or Lasso regression.
6. Report validation RMSE, MAE and R squared.
7. Interpret important coefficients, especially the fatigue and travel variables.
8. Save all Python code, result tables and figures.

## Member C task

Use `team_match_208.csv` to predict `team_goals`.

1. Exclude team names, opponent names, team IDs, opponent IDs and the target from the explanatory-variable matrix.
2. Use `match_id` only as the grouping variable for validation.
3. Keep both rows from the same match in the same training or validation fold.
4. Complete descriptive statistics, data visualisations and correlation analysis.
5. Check unusual values, multicollinearity and regression assumptions.
6. Build and evaluate a multiple linear regression model.
7. Compare it with at least one suitable alternative, such as Ridge or Lasso regression.
8. Report validation RMSE, MAE and R squared and save all code, tables and figures.

## Member D task

1. Prepare the four-page report structure.
2. Create one model-comparison table using the same metrics for both tasks.
3. Help select the final model for each regression task using validation performance.
4. Integrate the question, data method, results, discussion, limitations and conclusion.
5. Cite FIFA and FBref and explain the approximate travel measure.
6. Check the repository link, Turnitin report, signed AI Usage Declaration and final ZIP.
7. Maintain the contribution record and Microsoft Teams evidence.

## Rules that must not change

- All predictors must be available before the relevant match starts.
- Current-match performance data must not be used as predictors.
- Extra-time goals are included in goals; penalty shoot-out kicks are excluded.
- Cumulative extra time increases by 30 minutes only after an earlier extra-time match.
- Regression 2.2 validation must group observations by `match_id`.
- Both tasks should report RMSE, MAE and R squared to support comparison.

## Suggested internal dates

- 7 October: Members B and C share their first models and figures.
- 8 October: Cross-review and corrections.
- 9 October: Member D prepares the first integrated report draft.
- Before 10 October: Hold and record the Checkpoint 2 meeting.

## Package contents

- `data/raw`: retained source JSON and source manifest.
- `data/processed`: modelling datasets and supporting tables.
- `scripts`: reproducible download, preparation and validation scripts.
- `docs/member_A_data_dictionary.xlsx`: definitions, sources, units and timing rules.
- `docs/member_A_report_text.md`: concise material for the report's data-method section.
- `docs/CONTRIBUTIONS.md`: group contribution record to complete.
- `results`: validation checks and predictor lists.
- `checkpoint_1`: simple English Checkpoint 1 meeting script.

Run `python scripts/validate_data.py` from the package root to repeat the integrity audit.
