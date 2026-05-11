# MountDash ML Dataset Schema

This dataset trains the persisted ML pipeline used by the MountDash backend.

## Required columns

- `income`: Total monthly income.
- `total_expenses`: Total monthly expenses.
- `food_expense`: Monthly food and dining spend.
- `shopping_expense`: Monthly shopping spend.
- `rent_expense`: Monthly rent or housing spend.
- `savings`: `income - total_expenses` for that month.
- `remaining_balance`: End-of-period balance after the month closes.
- `overspending_label`: Binary target for logistic regression.
  - `1` = overspending risk present.
  - `0` = spending under control.

## Optional columns

- `month_index`: Sequential month number used by linear regression.
  If missing, the pipeline creates it automatically.

## Preprocessing rules

- CSV is loaded with `pandas`.
- Required columns must exist or training fails.
- Numeric fields are coerced to numeric values.
- Missing numeric values are filled with the column median.
- Rows with invalid `overspending_label` values are rejected.
- Derived fields are created automatically:
  - `savings_rate`
  - `spend_ratio`
  - `discretionary_spending_percentage`

## Model usage

- Logistic Regression predicts overspending risk.
- K-Means classifies spender type.
- Linear Regression predicts next-month expected expense.

## Replace dataset

To replace the dataset, overwrite:

`dataset/financial_behavior.csv`

Then retrain models with:

```powershell
npm run ml:train
```
