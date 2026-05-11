const path = require('path');
const { spawn } = require('child_process');
const Transaction = require('../models/Transaction');
const User = require('../models/User');

const DISCRETIONARY_CATEGORIES = new Set([
  'Shopping',
  'Entertainment',
  'Travel',
  'Food & Dining',
  'Other',
]);

const ESSENTIAL_CATEGORIES = new Set([
  'Bills & Utilities',
  'Healthcare',
  'Education',
  'Transportation',
]);

const ROOT_DIR = path.join(__dirname, '..', '..');
const DATASET_PATH = process.env.ML_DATASET_PATH || path.join(ROOT_DIR, 'dataset', 'financial_behavior.csv');
const ARTIFACT_DIR = process.env.ML_ARTIFACT_DIR || path.join(ROOT_DIR, 'models');

function resolvePythonCommandCandidates() {
  const candidates = [];
  const seen = new Set();

  const add = (value) => {
    if (!value) return;
    const normalized = String(value).trim();
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    candidates.push(normalized);
  };

  add(process.env.PYTHON_PATH);
  add(process.env.PYTHON_BIN);

  if (process.platform === 'win32') {
    add(path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe'));
    add('python');
    add('py');
    add('python3');
  } else {
    add(path.join(ROOT_DIR, '.venv', 'bin', 'python'));
    add('python3');
    add('python');
  }

  return candidates;
}

function monthKeyFromDate(dateInput) {
  const date = new Date(dateInput);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
}

function normalizeTransactionType(type) {
  const normalized = String(type || '').toLowerCase();
  if (normalized === 'income' || normalized === 'credit' || normalized === 'receive') return 'income';
  if (normalized === 'expense' || normalized === 'debit' || normalized === 'send' || normalized === 'withdraw') return 'expense';
  return 'transfer';
}

function buildEmptyMonth(month) {
  return {
    month,
    income: 0,
    expenses: 0,
    savings: 0,
    remainingBalance: 0,
    savingsRate: 0,
    expenseToIncomeRatio: 0,
    topCategoryShare: 0,
    discretionaryShare: 0,
    essentialShare: 0,
    transactionCount: 0,
    recurringExpenseAmount: 0,
    categoryTotals: {},
  };
}

function buildCategoryFeatureTotals(categoryTotals = {}) {
  return {
    foodExpense: Number(categoryTotals['Food & Dining'] || 0),
    shoppingExpense: Number(categoryTotals.Shopping || 0),
    rentExpense: Number(categoryTotals['Bills & Utilities'] || categoryTotals.Rent || 0),
  };
}

function finalizeMonthSummary(monthSummary, runningBalance, monthIndex = 1) {
  const categoryEntries = Object.entries(monthSummary.categoryTotals || {}).sort((a, b) => b[1] - a[1]);
  const topCategoryAmount = categoryEntries[0]?.[1] || 0;
  const discretionaryAmount = categoryEntries
    .filter(([category]) => DISCRETIONARY_CATEGORIES.has(category))
    .reduce((sum, [, amount]) => sum + amount, 0);
  const essentialAmount = categoryEntries
    .filter(([category]) => ESSENTIAL_CATEGORIES.has(category))
    .reduce((sum, [, amount]) => sum + amount, 0);

  const income = monthSummary.income;
  const expenses = monthSummary.expenses;
  const savings = income - expenses;
  const categoryFeatures = buildCategoryFeatureTotals(monthSummary.categoryTotals);
  const shoppingExpense = Number(categoryFeatures.shoppingExpense || 0);
  const transactionCount = Number(monthSummary.transactionCount || 0);
  const recurringExpenseBurden = expenses > 0 ? Number(monthSummary.recurringExpenseAmount || 0) / expenses : 0;
  const shoppingIntensity = transactionCount > 0 ? shoppingExpense / transactionCount : 0;
  const healthScoreRaw = (
    (income > 0 ? 100 : 40)
    + (Math.max(-0.2, Math.min(0.5, income > 0 ? savings / income : -0.2)) * 100)
    - (Math.max(0, (expenses > income && income > 0) ? ((expenses - income) / income) * 100 : 0))
    - (Math.max(0, discretionaryAmount > 0 && expenses > 0 ? (discretionaryAmount / expenses) * 40 : 0))
  );
  const financialHealthScore = Math.max(0, Math.min(100, Math.round(healthScoreRaw)));

  return {
    month: monthSummary.month,
    monthIndex,
    income,
    expenses,
    totalExpenses: expenses,
    savings,
    remainingBalance: runningBalance,
    savingsRate: income > 0 ? savings / income : 0,
    expenseToIncomeRatio: income > 0 ? expenses / income : expenses > 0 ? 1 : 0,
    topCategoryShare: expenses > 0 ? topCategoryAmount / expenses : 0,
    discretionaryShare: expenses > 0 ? discretionaryAmount / expenses : 0,
    essentialShare: expenses > 0 ? essentialAmount / expenses : 0,
    recurringExpenseBurden,
    shoppingIntensity,
    financialHealthScore,
    transactionCount,
    topCategory: categoryEntries[0]?.[0] || null,
    recurringExpenseAmount: Number(monthSummary.recurringExpenseAmount || 0),
    categoryTotals: monthSummary.categoryTotals,
    ...categoryFeatures,
  };
}

function buildMonthlyFeatureRows(transactions, openingBalance = 0) {
  const monthlyMap = new Map();

  for (const transaction of transactions) {
    const month = monthKeyFromDate(transaction.date);
    if (!month) continue;
    if (!monthlyMap.has(month)) {
      monthlyMap.set(month, buildEmptyMonth(month));
    }

    const bucket = monthlyMap.get(month);
    const amount = Math.abs(Number(transaction.amount || 0));
    const type = normalizeTransactionType(transaction.type);
    if (!Number.isFinite(amount) || amount <= 0) continue;

    if (type === 'income') {
      bucket.income += amount;
    } else if (type === 'expense') {
      bucket.expenses += amount;
      const category = transaction.category || 'Other';
      bucket.categoryTotals[category] = (bucket.categoryTotals[category] || 0) + amount;
      if (transaction.isRecurring) {
        bucket.recurringExpenseAmount += amount;
      }
    }

    bucket.transactionCount += 1;
  }

  const orderedMonths = Array.from(monthlyMap.keys()).sort();
  let runningBalance = Number(openingBalance || 0);

  return orderedMonths.map((month, index) => {
    const summary = monthlyMap.get(month);
    runningBalance += summary.income - summary.expenses;
    return finalizeMonthSummary(summary, runningBalance, index + 1);
  });
}

function buildCurrentFeatureRow(transactions, currentBalance = 0) {
  const currentMonth = monthKeyFromDate(new Date());
  const currentSummary = buildEmptyMonth(currentMonth);

  for (const transaction of transactions) {
    const amount = Math.abs(Number(transaction.amount || 0));
    const type = normalizeTransactionType(transaction.type);
    if (!Number.isFinite(amount) || amount <= 0) continue;

    if (type === 'income') {
      currentSummary.income += amount;
    } else if (type === 'expense') {
      currentSummary.expenses += amount;
      const category = transaction.category || 'Other';
      currentSummary.categoryTotals[category] = (currentSummary.categoryTotals[category] || 0) + amount;
      if (transaction.isRecurring) {
        currentSummary.recurringExpenseAmount += amount;
      }
    }

    currentSummary.transactionCount += 1;
  }

  return finalizeMonthSummary(currentSummary, Number(currentBalance || 0), 1);
}

function monthShift(monthKey, delta) {
  const [yearPart, monthPart] = String(monthKey).split('-');
  const date = new Date(Number(yearPart), Number(monthPart) - 1 + delta, 1);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
}

function ensureMinimumHistory(monthlyRows = [], minimum = 6) {
  if (monthlyRows.length >= minimum) return monthlyRows;
  if (!monthlyRows.length) return monthlyRows;

  const out = [...monthlyRows];
  const avgIncome = out.reduce((sum, row) => sum + Number(row.income || 0), 0) / out.length;
  const avgExpense = out.reduce((sum, row) => sum + Number(row.expenses || 0), 0) / out.length;
  const avgSavings = avgIncome - avgExpense;
  const firstMonth = out[0].month;
  const placeholdersNeeded = minimum - out.length;

  for (let index = placeholdersNeeded; index >= 1; index -= 1) {
    const month = monthShift(firstMonth, -index);
    out.unshift({
      ...buildEmptyMonth(month),
      month,
      monthIndex: 0,
      income: Math.max(0, Math.round(avgIncome * 0.92)),
      expenses: Math.max(0, Math.round(avgExpense * 0.92)),
      totalExpenses: Math.max(0, Math.round(avgExpense * 0.92)),
      savings: Math.round(avgSavings * 0.92),
      savingsRate: avgIncome > 0 ? avgSavings / avgIncome : 0,
      expenseToIncomeRatio: avgIncome > 0 ? avgExpense / avgIncome : 0,
      topCategory: 'Estimated',
      categoryTotals: {},
      isEstimated: true,
      financialHealthScore: 55,
    });
  }

  return out.map((row, index) => ({ ...row, monthIndex: index + 1 }));
}

function buildCategoryPressure(rows = []) {
  const totals = new Map();
  let totalExpenses = 0;

  for (const row of rows) {
    const categoryTotals = row.categoryTotals || {};
    for (const [category, amount] of Object.entries(categoryTotals)) {
      const numericAmount = Number(amount || 0);
      if (!Number.isFinite(numericAmount) || numericAmount <= 0) continue;
      totals.set(category, (totals.get(category) || 0) + numericAmount);
      totalExpenses += numericAmount;
    }
  }

  if (!totalExpenses) {
    return [];
  }

  return [...totals.entries()]
    .map(([category, amount]) => ({
      category,
      amount: Number(amount.toFixed(2)),
      share: Number(((amount / totalExpenses) * 100).toFixed(1)),
    }))
    .sort((a, b) => b.share - a.share);
}

function buildAnalyticsPayload(monthlyRows = [], ml = {}) {
  const rows = ensureMinimumHistory(monthlyRows, 6);
  const forecastBase = rows.length ? Number(rows[rows.length - 1].expenses || 0) : 0;
  const forecastValue = Number(ml?.predicted_expense || ml?.trend?.nextMonthExpense || forecastBase);
  const risk = String(ml?.overspending_risk || 'Low');
  const avgExpenses = rows.length
    ? rows.reduce((sum, row) => sum + Number(row.expenses || 0), 0) / rows.length
    : 0;
  const volatility = rows.length > 1
    ? Math.sqrt(
      rows.reduce((sum, row) => sum + ((Number(row.expenses || 0) - avgExpenses) ** 2), 0) / rows.length
    ) / (avgExpenses || 1)
    : 0;

  const anomalyHistory = rows.map((row, index) => {
    const current = Number(row.expenses || 0);
    const baselineSlice = rows.slice(Math.max(0, index - 3), index);
    const baseline = baselineSlice.length
      ? baselineSlice.reduce((sum, item) => sum + Number(item.expenses || 0), 0) / baselineSlice.length
      : avgExpenses;
    const anomalyScore = baseline > 0 ? (current - baseline) / baseline : 0;
    return {
      month: row.month,
      anomaly: anomalyScore > 0.3,
      anomaly_score: Number(anomalyScore.toFixed(4)),
      expense_delta: baseline > 0 ? (current - baseline) / baseline : 0,
    };
  });

  return {
    monthlyBehaviorRows: rows.map((row) => ({
      month: row.month,
      total_income: Number(row.income || 0),
      total_expenses: Number(row.expenses || 0),
      savings: Number(row.savings || 0),
      savings_rate: Number(row.savingsRate || 0),
      shopping_ratio: Number(row.totalExpenses > 0 ? (row.shoppingExpense || 0) / row.totalExpenses : 0),
      food_ratio: Number(row.totalExpenses > 0 ? (row.foodExpense || 0) / row.totalExpenses : 0),
      recurring_expense_burden: Number(row.recurringExpenseBurden || 0),
      discretionary_spending: Number(row.discretionaryShare || 0),
      spending_volatility: Number(row.spendingVolatility || volatility || 0),
      health_score: Number(row.financialHealthScore || 0),
      overspending_risk: Number(row.expenseToIncomeRatio || 0) >= 0.9 ? 'High' : Number(row.expenseToIncomeRatio || 0) >= 0.75 ? 'Medium' : risk,
      anomaly_score: Number(anomalyHistory.find((entry) => entry.month === row.month)?.anomaly_score || 0),
      anomaly: Boolean(anomalyHistory.find((entry) => entry.month === row.month)?.anomaly),
      estimated: Boolean(row.isEstimated),
    })),
    forecastChart: rows.map((row, index) => ({
      month: row.month,
      actual_expense: Number(row.expenses || 0),
      forecast_expense: index === rows.length - 1 ? forecastValue : Number(row.expenses || 0),
    })),
    healthTrend: rows.map((row) => ({
      month: row.month,
      health_score: Number(row.financialHealthScore || 0),
      savings_rate: Number(row.savingsRate || 0),
    })),
    categoryPressure: buildCategoryPressure(rows),
    anomalyHistory,
  };
}

function spawnPythonCandidate(pythonCommand, command, payload = null) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '..', 'ml', 'pipeline.py');
    const child = spawn(pythonCommand, [scriptPath, command, '--dataset', DATASET_PATH, '--artifacts', ARTIFACT_DIR], {
      cwd: ROOT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (error) => {
      const enriched = new Error(`Failed to start ML pipeline with ${pythonCommand}: ${error.message}`);
      enriched.code = error.code;
      enriched.pythonCommand = pythonCommand;
      enriched.scriptPath = scriptPath;
      reject(enriched);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        const enriched = new Error(stderr.trim() || `Python ML pipeline exited with code ${code}`);
        enriched.code = code;
        enriched.pythonCommand = pythonCommand;
        enriched.scriptPath = scriptPath;
        reject(enriched);
        return;
      }

      try {
        resolve({
          pythonCommand,
          scriptPath,
          payload: JSON.parse(stdout || '{}'),
        });
      } catch (error) {
        const enriched = new Error(`Unable to parse ML pipeline output from ${pythonCommand}: ${error.message}`);
        enriched.pythonCommand = pythonCommand;
        enriched.scriptPath = scriptPath;
        reject(enriched);
      }
    });

    if (payload) {
      child.stdin.write(JSON.stringify(payload));
    }
    child.stdin.end();
  });
}

async function runPythonMlCommand(command, payload = null) {
  const candidates = resolvePythonCommandCandidates();
  const failures = [];

  for (const candidate of candidates) {
    try {
      const result = await spawnPythonCandidate(candidate, command, payload);
      return {
        ...result.payload,
        pythonCommand: result.pythonCommand,
      };
    } catch (error) {
      failures.push({
        pythonCommand: candidate,
        message: error.message,
        code: error.code || null,
      });

      const isMissingBinary = error.code === 'ENOENT';
      if (!isMissingBinary) {
        error.attemptedCommands = candidates;
        error.failures = failures;
        throw error;
      }
    }
  }

  const summary = failures.map((failure) => `${failure.pythonCommand}: ${failure.message}`).join(' | ');
  const finalError = new Error(`Unable to find a working Python runtime for ML pipeline. Attempts: ${summary}`);
  finalError.attemptedCommands = candidates;
  finalError.failures = failures;
  throw finalError;
}

async function checkMlRuntime() {
  try {
    const status = await runPythonMlCommand('runtime');
    return {
      ...status,
      attemptedCommands: resolvePythonCommandCandidates(),
    };
  } catch (error) {
    return {
      ready: false,
      pythonCommand: process.env.PYTHON_PATH || process.env.PYTHON_BIN || 'python3',
      scriptPath: path.join(__dirname, '..', 'ml', 'pipeline.py'),
      sklearn: false,
      pandas: false,
      joblib: false,
      datasetExists: false,
      datasetPath: DATASET_PATH,
      artifactDir: ARTIFACT_DIR,
      artifactsReady: false,
      attemptedCommands: resolvePythonCommandCandidates(),
      failures: error.failures || [],
      error: error.message,
    };
  }
}

function buildRuleBasedMlFallback(monthlyRows, currentRow) {
  const avgExpenses = monthlyRows.length
    ? monthlyRows.reduce((sum, row) => sum + row.expenses, 0) / monthlyRows.length
    : currentRow.expenses;
  const overspending = currentRow.expenses > currentRow.income || currentRow.savings < 0;
  const nextMonthExpense = avgExpenses;

  let segment = 'Moderate Spender';
  if (currentRow.expenseToIncomeRatio >= 0.85 || currentRow.discretionaryShare >= 0.45) {
    segment = 'High Spender';
  } else if (currentRow.expenseToIncomeRatio <= 0.45 && currentRow.savingsRate >= 0.3) {
    segment = 'Low Spender';
  }

  return {
    source: 'fallback_rules',
    overspending_risk: overspending ? 'High' : 'Low',
    confidence: overspending ? 70 : 25,
    predicted_expense: Math.round(nextMonthExpense),
    spender_type: segment,
    overspending: {
      prediction: overspending ? 'High overspending risk' : 'Low overspending risk',
      probability: overspending ? 0.7 : 0.25,
      confidence: overspending ? 70 : 25,
      riskLevel: overspending ? 'high' : 'low',
      model: 'Rule-based fallback',
      metrics: {},
    },
    behavior: {
      segment,
      cluster: segment === 'High Spender' ? 2 : segment === 'Low Spender' ? 0 : 1,
      model: 'Rule-based fallback',
      metrics: {},
    },
    trend: {
      nextMonthExpense: Math.round(nextMonthExpense),
      direction: 'stable',
      confidenceScore: 0,
      model: 'Rule-based fallback',
      metrics: {},
    },
    training: {
      status: 'fallback',
      sampleCount: monthlyRows.length,
      invalidRowsDropped: 0,
      datasetPath: DATASET_PATH,
      artifactDir: ARTIFACT_DIR,
    },
    analytics: {
      monthlyHealthScore: Number(currentRow.financialHealthScore || 0),
      spendingVolatility: 0,
      anomaly: false,
    },
  };
}

function addBehaviorSignals(monthlyRows = [], currentRow = {}) {
  if (!monthlyRows.length) {
    return {
      rows: monthlyRows,
      current: {
        ...currentRow,
        spendingVolatility: 0,
        anomalyScore: 0,
        anomaly: false,
      },
      spendingVolatility: 0,
      anomalyScore: 0,
      anomaly: false,
    };
  }

  const expenses = monthlyRows.map((row) => Number(row.expenses || 0));
  const mean = expenses.reduce((sum, value) => sum + value, 0) / expenses.length;
  const variance = expenses.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / Math.max(1, expenses.length);
  const stdDev = Math.sqrt(variance);
  const spendingVolatility = mean > 0 ? stdDev / mean : 0;
  const baseline = expenses.length > 1 ? expenses.slice(0, -1) : expenses;
  const baselineMean = baseline.reduce((sum, value) => sum + value, 0) / Math.max(1, baseline.length);
  const latestExpense = Number(monthlyRows[monthlyRows.length - 1]?.expenses || 0);
  const anomalyScore = baselineMean > 0 ? (latestExpense - baselineMean) / baselineMean : 0;
  const anomaly = anomalyScore > 0.3;

  const rows = monthlyRows.map((row) => ({
    ...row,
    spendingVolatility,
  }));

  return {
    rows,
    current: {
      ...currentRow,
      spendingVolatility,
      anomalyScore,
      anomaly,
    },
    spendingVolatility,
    anomalyScore,
    anomaly,
  };
}

async function trainMlModels() {
  try {
    return await runPythonMlCommand('train');
  } catch (error) {
    console.error('[ML] Training failed:', {
      message: error.message,
      failures: error.failures || [],
      attemptedCommands: error.attemptedCommands || resolvePythonCommandCandidates(),
    });
    throw error;
  }
}

async function getMlInsightsForUser(userId) {
  const [transactions, user] = await Promise.all([
    Transaction.find({ user: userId }).sort({ date: 1, createdAt: 1 }).lean(),
    User.findById(userId).select('balance').lean(),
  ]);

  const runningBalance = Number(user?.balance || 0);
  const monthlyRows = buildMonthlyFeatureRows(transactions, 0);
  const currentRow = monthlyRows.length
    ? {
      ...monthlyRows[monthlyRows.length - 1],
      remainingBalance: runningBalance,
    }
    : buildCurrentFeatureRow(transactions, runningBalance);
  const behaviorSignals = addBehaviorSignals(monthlyRows, currentRow);
  const enrichedMonthlyRows = behaviorSignals.rows;
  const enrichedCurrentRow = behaviorSignals.current;

  if (!monthlyRows.length) {
    const fallbackMl = buildRuleBasedMlFallback([], enrichedCurrentRow);
    return {
      monthlyRows: [],
      currentRow: enrichedCurrentRow,
      ml: fallbackMl,
      analytics: buildAnalyticsPayload([], fallbackMl),
    };
  }

  try {
    const ml = await runPythonMlCommand('predict', {
      monthlyHistory: monthlyRows,
      current: enrichedCurrentRow,
    });

    return {
      monthlyRows: enrichedMonthlyRows,
      currentRow: enrichedCurrentRow,
      ml: {
        source: 'trained_model',
        analytics: {
          monthlyHealthScore: Number(enrichedCurrentRow.financialHealthScore || 0),
          spendingVolatility: behaviorSignals.spendingVolatility,
          anomaly: behaviorSignals.anomaly,
          anomalyScore: behaviorSignals.anomalyScore,
        },
        ...ml,
      },
      analytics: buildAnalyticsPayload(enrichedMonthlyRows, ml),
    };
  } catch (error) {
    console.error('ML pipeline fallback triggered:', {
      message: error.message,
      failures: error.failures || [],
      attemptedCommands: error.attemptedCommands || resolvePythonCommandCandidates(),
    });
    const fallbackMl = {
      ...buildRuleBasedMlFallback(enrichedMonthlyRows, enrichedCurrentRow),
      analytics: {
        monthlyHealthScore: Number(enrichedCurrentRow.financialHealthScore || 0),
        spendingVolatility: behaviorSignals.spendingVolatility,
        anomaly: behaviorSignals.anomaly,
        anomalyScore: behaviorSignals.anomalyScore,
      },
    };
    return {
      monthlyRows: enrichedMonthlyRows,
      currentRow: enrichedCurrentRow,
      ml: fallbackMl,
      analytics: buildAnalyticsPayload(enrichedMonthlyRows, fallbackMl),
    };
  }
}

module.exports = {
  buildMonthlyFeatureRows,
  buildCurrentFeatureRow,
  getMlInsightsForUser,
  checkMlRuntime,
  trainMlModels,
  resolvePythonCommandCandidates,
  DATASET_PATH,
  ARTIFACT_DIR,
};

