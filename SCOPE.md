# Scope

## Solution (one-liner)
A web app where a user uploads a customer file and gets back each customer's
churn probability and risk band, the columns that drive churn overall, and a
recommended retention action for the highest-risk customers. The model used for
scoring is chosen automatically by a fixed rule: logistic regression, unless a
challenger beats both baselines by an agreed margin.

## In Scope
- The public Telco customer churn dataset (downloaded by hand, never committed), a
  synthetic data generator, and deliberately broken files for the upload checker's tests
- CSV upload with validation and plain-language errors
- Two modes: train and score (file has a churn column), or score only (it does not)
- Cleaning and preprocessing with one pipeline reused at training and prediction time
- Four models, including two baselines, compared in one training run with the same
  cross-validation folds, then one check on held-back rows
- Model comparison (PR-AUC, ROC-AUC, recall in the top 10%, Brier score) and automatic
  selection of the scoring model by the margin rule, with the leaderboard shown
- Churn probability for a whole file, with a High, Medium or Low risk band
- Global ranking of the columns that drive churn, with its stability across seeds
- Rule-based retention actions with a stated reason for every High-risk customer
- A drift warning when a scored file looks different from the training data
- Downloadable scored file (ID, probability, risk band, action, reason)
- A "Try a sample file" button, and a demo mode where each visitor trains a private model
- Containerized app (Docker, Docker Compose), automated checks in CI, and a live demo
  on Render
- A README and a model card

## Out of Scope
- Real customer data or real personal information
- Datasets other than Telco-shaped files (the bank dataset is not supported)
- Multi-tenant accounts, billing, any user management, or a login
- Database or CRM connectors (file upload only)
- Uplift modelling (needs campaign outcome data that does not exist yet)
- Streaming or real-time ingestion
- Mobile app
- Excel or Parquet upload, and dataset versioning
- Automatic exploratory report
- Feature engineering with leakage checks
- Training as a background job (it runs while the screen waits)
- ROC, precision-recall and lift charts, and probability calibration (the screen warns
  that probabilities run high; risk bands use the ranking only)
- Single-customer form
- Per-customer explanations
- Ranking actions by expected value (needs assumed offer cost and success rate)
- MLflow, a database, and extra libraries from the original plan (SHAP, XGBoost,
  LightGBM, Optuna, pandera, Plotly)

## Done for v1
A working app with the features above, passing automated checks, running as a public
demo, with a README and a model card.
