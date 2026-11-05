# Model card: customer churn scorer

A model card is a short, honest description of a model: what it is for, what it learned
from, how well it works, and where it should not be trusted. This one covers the model
the app chooses on the Telco dataset. Numbers were measured with the code in this
repository (seed 42) and will change a little with another random split.

## Model details

- **Task:** estimate how likely each customer is to churn (leave), and rank customers so
  the riskiest can be contacted first.
- **Model:** logistic regression (scikit-learn) with balanced class weights, inside one
  pipeline that cleans the raw columns, one-hot encodes categories, scales numbers and
  then predicts. The same pipeline object is used for training and scoring.
- **How it was chosen:** four models were compared with 5-fold stratified
  cross-validation on the training rows: a majority-class guess, logistic regression,
  random forest and gradient boosting. A challenger replaces logistic regression only if
  it beats both baselines by at least 0.02 PR-AUC. None did on Telco, so logistic
  regression was kept.
- **Reproducibility:** fixed seed 42, exact library versions in `uv.lock`.

## Intended use

- **For:** ranking the customers of a telecom-style subscription business to decide who a
  retention team should contact first, and a portfolio demonstration of a careful
  machine-learning workflow.
- **Not for:** decisions about individual people with legal or financial consequences, any
  use on data with different columns, or use as a calibrated probability of leaving.
  Nothing here has been tested on real customers of a real company.

## Data

- **Training data:** the public "Telco Customer Churn" dataset (Kaggle, based on IBM sample
  data): 7,043 customers, 21 columns, about 26.5% churned. It is not included in this
  repository. Check its page for its terms before reusing it.
- **Split:** a stratified random 80/20 split (5,634 training and 1,409 held-back customers).
  The data has no dates, so a split by time was not possible. Scores on the held-back rows
  may therefore look better than future performance.
- **Cleaning:** blank `TotalCharges` cells (customers with no billing yet) are set to 0,
  and the yes/no columns become 1 and 0.
- **Inputs used:** 19 of the 21 columns: services and contract (for example `Contract`,
  `InternetService`, `TechSupport`), billing (`MonthlyCharges`, `TotalCharges`,
  `PaymentMethod`, `PaperlessBilling`), `tenure`, and personal attributes (`gender`,
  `SeniorCitizen`, `Partner`, `Dependents`). The customer ID is never a model input.

## Results

PR-AUC (average precision) is the main metric because churn is the minority class.
A model that guesses at random scores about the churn rate (0.265).

**Cross-validation on the training rows (mean of 5 folds, ± spread across folds):**

| Model | PR-AUC | ROC-AUC |
| --- | --- | --- |
| Majority-class guess | 0.265 | 0.500 |
| **Logistic regression (chosen)** | **0.660 ± 0.020** | **0.846 ± 0.012** |
| Random forest | 0.653 ± 0.022 | 0.845 ± 0.010 |
| Gradient boosting | 0.648 ± 0.020 | 0.838 ± 0.008 |

The differences between the three real models are smaller than the spread across folds,
so they cannot be told apart reliably.

**One check on the held-back 20% (1,409 customers, 374 of whom churned):**

| Metric | Value | What it means |
| --- | --- | --- |
| PR-AUC | 0.633 | Ranking quality for the churners; 0.265 is guessing |
| ROC-AUC | 0.842 | Chance a churner is ranked above a loyal customer |
| Recall in the riskiest 10% | 0.286 | About 107 of the 374 churners are among the 141 customers ranked riskiest |
| Brier score | 0.169 | Gap between predicted chance and outcome (0 is perfect). Always predicting the 26.5% churn rate scores about 0.195 |

The held-back rows were looked at once and nothing was changed because of them.

## What drives the predictions

Permutation importance on the held-back rows (how much PR-AUC drops when one column is
shuffled): `tenure` (0.227), `InternetService` (0.086), `MonthlyCharges` (0.073),
`Contract` (0.044) and `TotalCharges` (0.027). These five were identical and in the same
order for three different random seeds. Ranks 6 to 10 changed between seeds (7 of the
top 10 were shared), and their scores are small, so their order is not reliable.
Importance says which columns the model relies on, not whether they raise or lower the
risk. `tenure` and `TotalCharges` are closely related, which can split credit between them.

## Risk bands and actions

- **Bands** are set by rank: the riskiest 10% of a file (adjustable) are High, the next 20%
  are Medium, and the rest Low. Every file therefore has a High group, even if few of those
  customers are likely to leave.
- **Actions** come from a short list of hand-written rules (new customer, fibre internet
  without tech support, pays by electronic check, month-to-month contract), checked from
  the most specific to the least, with a default. They are business suggestions. Nothing in
  this project shows that they keep customers.

## Limitations

- **Probabilities are not calibrated.** Balanced class weights push them upward (in an earlier
  check the average predicted probability on the held-back rows was 0.416, against a true
  churn rate of 0.265). Use them to
  rank, not as chances.
- **Telco-shaped files only.** The model expects the Telco column names and values.
- **One dataset, one split.** There is no second dataset and no split by time, so the
  results say little about other businesses or about performance next year.
- **No tuning.** The random forest and gradient boosting settings are sensible defaults. A
  tuned challenger might do better.
- **Drift warnings use rules of thumb.** PSI cutoffs of 0.1 and 0.25 were checked only on
  Telco, and small files (under 200 rows) are not checked.

## Ethical considerations

- `gender` and `SeniorCitizen` are model inputs, and `Partner` and `Dependents` describe a
  person's circumstances. On the held-back rows, shuffling `gender` and `SeniorCitizen`
  changed PR-AUC by almost nothing (0.0005 and -0.0015), but this is not a fairness audit,
  and other columns can act as stand-ins for protected attributes. A real deployment
  should review which inputs are acceptable and test results across groups.
- The app asks users not to upload real personal data, and the demo stores nothing about
  customers: a saved model holds learned numbers, and the drift reference holds only group
  edges and shares.

## The model in the live demo

The hosted demo ships with a different model, trained at build time on made-up
customers from `src/churn_platform/synthetic.py`, so that it works before anyone trains
anything. Its scores are not Telco results. In the demo, each visitor's Train creates a
private model for their session only.

## Monitoring

When a file is scored, each input column is compared with how it was spread in the
training rows, using the population stability index, and a warning appears if the shift
is moderate or major. It says the file looks different, not that the scores are wrong.
