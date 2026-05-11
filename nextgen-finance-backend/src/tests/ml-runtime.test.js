const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');

process.env.PYTHON_PATH = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
process.env.ML_DATASET_PATH = path.join(__dirname, '..', '..', 'dataset', 'financial_behavior.csv');
process.env.ML_ARTIFACT_DIR = path.join(__dirname, '..', '..', 'models');

const { checkMlRuntime, trainMlModels, resolvePythonCommandCandidates } = require('../services/mlInsightsService');

test('resolvePythonCommandCandidates prefers configured runtime first', () => {
  const candidates = resolvePythonCommandCandidates();
  assert.equal(candidates[0], process.env.PYTHON_PATH);
  assert.ok(candidates.includes('python'));
});

test('checkMlRuntime returns a structured status object', async () => {
  await trainMlModels();
  const status = await checkMlRuntime();

  assert.equal(typeof status.ready, 'boolean');
  assert.equal(typeof status.sklearn, 'boolean');
  assert.equal(typeof status.pythonCommand, 'string');
  assert.equal(typeof status.scriptPath, 'string');
  assert.equal(typeof status.datasetExists, 'boolean');
  assert.equal(typeof status.artifactsReady, 'boolean');
  assert.equal(status.datasetExists, true);
});
