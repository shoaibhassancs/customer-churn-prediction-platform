"""Customer churn prediction platform.

Modules, in the order the data flows through them:

1. cleaning       fixed rules that fix columns and encode the target and flags
2. splitting      stratified random train/test split
3. preprocessing  one-hot, scaling and passthrough (learns from training rows)
4. pipeline       chains cleaning and preprocessing into one reusable object
"""
