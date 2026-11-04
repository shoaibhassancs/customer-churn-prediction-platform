# Scope

## Solution (one-liner)
A web app where a user uploads a customer file and gets back each customer's
churn probability, the main reasons behind it, and a recommended retention action.
The model used for scoring is chosen automatically by a fixed rule: logistic
regression, unless a challenger beats both baselines by an agreed margin.

## In Scope
- Public datasets (telecom, bank), a synthetic data generator, and deliberately
  broken files for validator tests
- File upload (CSV, Excel, Parquet) with validation, versioning and plain-language
  errors
- Two modes: train and score (file has a churn column), or score only (it does not)
- Cleaning and preprocessing with one pipeline reused at training and prediction
  time
- Automatic exploratory report (churn rate, distributions, segments, missingness)
- Feature engineering with a leakage check for every feature
- At least four models, including a baseline, trained in one background job
- Model comparison (PR-AUC, ROC-AUC, lift, calibration) and automatic selection of
  the scoring model by the margin rule, with the leaderboard shown
- Churn probability for one customer or a whole file, with risk band
- Global and per-customer explanations of what drives churn
- Rule-based retention recommendations, ranked by expected value
- Downloadable scored file (ID, probability, risk band, top three drivers, action)
- Containerized app, automated checks in CI, a drift alert, and one simple login
- A live demo with sample data, a README and a model card

## Out of Scope
- Real customer data or real personal information
- Multi-tenant accounts, billing or user management beyond one simple login
- Database or CRM connectors (file upload only)
- Uplift modelling (needs campaign outcome data that does not exist yet)
- Streaming or real-time ingestion
- Mobile app

## Done for v1
All six phase gates in `project-notes/PLAN.md` pass, ending with a public demo.
