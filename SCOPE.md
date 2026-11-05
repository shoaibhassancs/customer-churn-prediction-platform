# Scope

## Solution (one-liner)
A web app where a user uploads a customer file and gets back each customer's
churn probability and risk band, the columns that drive churn overall, and a
recommended retention action for the highest-risk customers.
The model used for scoring is chosen automatically by a fixed rule: logistic
regression, unless a challenger beats both baselines by an agreed margin.

## In Scope
- Public datasets (telecom, bank), a synthetic data generator, and deliberately
  broken files for validator tests
- CSV upload with validation and plain-language errors
- Two modes: train and score (file has a churn column), or score only (it does not)
- Cleaning and preprocessing with one pipeline reused at training and prediction
  time
- At least four models, including a baseline, compared in one training run
- Model comparison (PR-AUC, ROC-AUC, recall in the top 10%, Brier score) and
  automatic selection of the scoring model by the margin rule, with the leaderboard
  shown
- Churn probability for a whole file, with risk band
- Global ranking of the columns that drive churn
- Rule-based retention actions with a stated reason for every High-risk customer
- Downloadable scored file (ID, probability, risk band, action, reason)
- Containerized app, automated checks in CI, a drift alert, and one simple login
- A live demo with sample data, a README and a model card

## Deferred (not needed for v1; can be brought back later)
Decided after Phase 4: no phase gate needs these, so they are out of v1 unless
reopened.
- Excel and Parquet upload, and dataset versioning (CSV only for now)
- Automatic exploratory report
- Feature engineering with a leakage check for every feature
- Training as a background job (it runs while the screen waits)
- Lift charts, ROC and precision-recall curves, and probability calibration
  (the screen warns that probabilities run high; bands use the ranking only)
- Single-customer form
- Per-customer explanations and top-three drivers per customer in the scored file
- Ranking actions by expected value (needs assumed offer cost and success rate)

## Out of Scope
- Real customer data or real personal information
- Multi-tenant accounts, billing or user management beyond one simple login
- Database or CRM connectors (file upload only)
- Uplift modelling (needs campaign outcome data that does not exist yet)
- Streaming or real-time ingestion
- Mobile app

## Done for v1
All six phase gates in `project-notes/PLAN.md` pass, ending with a public demo.
