# Contributions — HIT140 Assessment 3, Group 19

**Unit:** HIT140 Foundations of Data Science
**Campus:** Darwin
**Report:** Predicting Match Outcomes at the FIFA World Cup 2026
**Submitted:** 15 October 2026

---

## Members and roles

| Member | Name | Student ID | Role | Sections owned |
|---|---|---|---|---|
| A | Haonan Ding | S394323 | Data collection and dataset construction | 2.1, 2.2, 2.3, 2.4 |
| B | Imelda Cherono | S404388 | Regression 2.1 — goal difference | 3.1, 3.2, 3.3, 3.4 |
| C | Shishir Chuke | S407059 | Regression 2.2 — team goals | 4.1, 4.2, 4.3, 4.4 |
| D | Prakash Subedi | S407192 | Model comparison, report integration, submission | 1, 5, 6, 7 |

---

## What each member did

### Haonan Ding (S394323) — Member A
- Collected match, ranking, fixture and base-camp data from FIFA's public data services
- Cross-checked fixtures against FBref
- Built both analysis-ready datasets (104 match rows; 208 team-match rows)
- Engineered all pre-match predictors: rest days, travel distance (Haversine), cumulative extra time, form and congestion measures
- Ran the quality checks confirming no missing model values and reconciled goals against the official match registry
- Produced the data dictionary used for the Section 2.3 variable table
- Wrote Section 2

### Imelda Cherono (S404388) — Member B
- Built and evaluated Regression 2.1 (target: Team A goal difference)
- Compared OLS, Ridge and Lasso on an 80:20 split and by repeated five-fold cross-validation
- Ran the VIF and collinearity checks (max 3.22)
- Produced Figure 1 (standardised effects; actual vs held-out predictions)
- Wrote Section 3

### Shishir Chuke (S407059) — Member C
- Built and evaluated Regression 2.2 (target: team goals)
- Implemented the grouped train/validation split by `match_id` to keep both rows of a match together
- Tuned Ridge and Lasso by grouped five-fold cross-validation
- Produced Figure 2 (actual vs predicted; residuals)
- Wrote Section 4

### Prakash Subedi (S407192) — Member D
- Compared the two models against their own baselines and built Table 5.1
- Wrote Sections 1, 5, 6 and 7
- Integrated all members' sections into the final report and managed the word and page budgets
- Set up and maintained the GitHub repository
- Ran the final consistency checks (figures, tables, terminology, numbering)
- Managed submission, this contribution record and the AI usage declaration

---

## Meetings

| # | Date | Platform | Attended | Purpose |
|---|---|---|---|---|
| 1 | <!-- FILL date --> | Microsoft Teams | <!-- FILL names --> | Task allocation and dataset scope |
| 2 | <!-- FILL date --> | Microsoft Teams | <!-- FILL names --> | Checkpoint 1 — progress review |
| 3 | <!-- FILL date --> | Microsoft Teams | <!-- FILL names --> | Results review and report structure |
| 4 | <!-- FILL date --> | Microsoft Teams | <!-- FILL names --> | Final review before submission |

Recordings and chat records: <!-- FILL Teams links -->

---

## AI usage

All members completed and signed the unit's AI Usage Declaration Form, which is
submitted alongside this report. AI assistance was used and has been declared.

---

## Sign-off

Each member confirms the record above is accurate and that they can explain
their own section.

- [ ] Haonan Ding (S394323) — contributions accurate
- [ ] Imelda Cherono (S404388) — contributions accurate
- [ ] Shishir Chuke (S407059) — contributions accurate
- [ ] Prakash Subedi (S407192) — contributions accurate
- [ ] All four AI declaration forms signed
- [ ] All code pushed to this repository
- [ ] GitHub link included in the report
- [ ] Final PDF checked against the page limit
- [ ] Submitted to Learnline, receipt saved
