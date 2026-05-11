const test = require('node:test');
const assert = require('node:assert/strict');

const { buildImportPreview, buildMlDatasetPreview } = require('../services/importPreviewService');
const { detectMlDatasetSchema } = require('../services/mlDatasetService');

test('buildImportPreview separates accepted and skipped rows', () => {
  const preview = buildImportPreview([
    { amount: '1000', type: 'income', category: 'salary', description: 'Monthly salary', date: '2026-04-01' },
    { amount: '0', type: 'expense', category: 'food', description: 'Bad row', date: '2026-04-02' },
  ]);

  assert.ok(preview.previewId);
  assert.equal(preview.previewType, 'transactions');
  assert.equal(preview.summary.totalRows, 2);
  assert.equal(preview.summary.acceptedRows, 1);
  assert.equal(preview.summary.skippedRows, 1);
  assert.equal(preview.acceptedRows[0].normalized.type, 'income');
  assert.equal(preview.skippedRows[0].reason, 'Amount must be greater than zero');
});

test('buildImportPreview counts inferred categories', () => {
  const preview = buildImportPreview([
    { amount: '450', type: 'expense', category: 'misc', description: 'Zomato order', date: '2026-03-15' },
  ]);

  assert.equal(preview.summary.acceptedRows, 1);
  assert.equal(preview.summary.inferredCategories, 1);
});

test('detectMlDatasetSchema accepts trimmed schema headers', () => {
  const rows = [{
    ' income ': '50000',
    'total_expenses ': '32000',
    ' food_expense': '6000',
    shopping_expense: '5000',
    rent_expense: '12000',
    savings: '18000',
    remaining_balance: '30000',
    overspending_label: '0',
    month_index: '1',
  }];

  assert.equal(detectMlDatasetSchema(rows), true);
});

test('buildMlDatasetPreview validates numeric rows and accepts string labels', () => {
  const preview = buildMlDatasetPreview([
    {
      income: '50000',
      total_expenses: '32000',
      food_expense: '6000',
      shopping_expense: '5000',
      rent_expense: '12000',
      savings: '18000',
      remaining_balance: '30000',
      overspending_label: '1',
      month_index: '2',
    },
    {
      income: 'oops',
      total_expenses: '32000',
      food_expense: '6000',
      shopping_expense: '5000',
      rent_expense: '12000',
      savings: '18000',
      remaining_balance: '30000',
      overspending_label: 'maybe',
      month_index: '0',
    },
  ]);

  assert.equal(preview.previewType, 'ml_dataset');
  assert.equal(preview.summary.acceptedRows, 1);
  assert.equal(preview.summary.skippedRows, 1);
  assert.equal(preview.acceptedRows[0].normalized.overspending_label, 1);
  assert.match(preview.skippedRows[0].reason, /income must be a valid number/);
  assert.match(preview.skippedRows[0].reason, /overspending_label must be 0 or 1/);
});
