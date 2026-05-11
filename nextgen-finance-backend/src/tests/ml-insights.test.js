const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');

process.env.PYTHON_PATH = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
process.env.ML_DATASET_PATH = path.join(__dirname, '..', '..', 'dataset', 'financial_behavior.csv');
process.env.ML_ARTIFACT_DIR = path.join(__dirname, '..', '..', 'models');

const { buildMonthlyFeatureRows, trainMlModels } = require('../services/mlInsightsService');

test('buildMonthlyFeatureRows creates monthly training data from transactions', () => {
  const rows = buildMonthlyFeatureRows([
    {
      type: 'income',
      amount: 50000,
      date: '2026-01-05T00:00:00.000Z',
    },
    {
      type: 'expense',
      amount: 15000,
      category: 'Bills & Utilities',
      date: '2026-01-15T00:00:00.000Z',
    },
    {
      type: 'expense',
      amount: 5000,
      category: 'Shopping',
      date: '2026-02-05T00:00:00.000Z',
    },
    {
      type: 'expense',
      amount: 2200,
      category: 'Food & Dining',
      date: '2026-02-10T00:00:00.000Z',
    },
    {
      type: 'income',
      amount: 52000,
      date: '2026-02-12T00:00:00.000Z',
    },
  ]);

  assert.equal(rows.length, 2);
  assert.equal(rows[0].month, '2026-01');
  assert.equal(rows[0].income, 50000);
  assert.equal(rows[0].expenses, 15000);
  assert.equal(rows[0].savings, 35000);
  assert.equal(rows[0].rentExpense, 15000);
  assert.equal(rows[1].month, '2026-02');
  assert.equal(rows[1].topCategory, 'Shopping');
  assert.equal(rows[1].foodExpense, 2200);
  assert.ok(rows[1].expenseToIncomeRatio > 0);
  assert.equal(rows[1].monthIndex, 2);
});

test('trainMlModels returns persisted metrics from the CSV dataset', async () => {
  const result = await trainMlModels();

  assert.equal(result.status, 'trained');
  assert.ok(result.dataset.row_count >= 10);
  assert.equal(typeof result.models.logistic_regression.accuracy, 'number');
  assert.equal(typeof result.models.logistic_regression.precision, 'number');
  assert.equal(typeof result.models.logistic_regression.recall, 'number');
  assert.equal(typeof result.models.logistic_regression.f1_score, 'number');
});
