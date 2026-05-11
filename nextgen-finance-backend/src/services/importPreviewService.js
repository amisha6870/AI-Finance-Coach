const crypto = require('crypto');
const TransactionService = require('./transactionService');
const { inspectMlDatasetRow, REQUIRED_COLUMNS } = require('./mlDatasetService');

const previewStore = new Map();
const PREVIEW_TTL_MS = 15 * 60 * 1000;

function cleanupExpiredPreviews() {
  const now = Date.now();
  for (const [key, value] of previewStore.entries()) {
    if (value.expiresAt <= now) {
      previewStore.delete(key);
    }
  }
}

function savePreview(type, payload) {
  const previewId = crypto.randomUUID();
  cleanupExpiredPreviews();
  previewStore.set(previewId, {
    ...payload,
    type,
    previewType: type,
    createdAt: Date.now(),
    expiresAt: Date.now() + PREVIEW_TTL_MS,
  });

  return {
    previewId,
    previewType: type,
    ...payload,
  };
}

function buildImportPreview(rows = []) {
  const acceptedRows = [];
  const skippedRows = [];
  let inferredCategoryCount = 0;

  rows.forEach((row, index) => {
    const result = TransactionService.inspectImportedTransaction(row);
    if (!result.accepted) {
      skippedRows.push({
        rowNumber: index + 2,
        reason: result.reason,
        reasons: [result.reason],
        raw: row,
      });
      return;
    }

    if (result.inferredCategory) {
      inferredCategoryCount += 1;
    }

    acceptedRows.push({
      rowNumber: index + 2,
      raw: row,
      normalized: result.transaction,
      inferredCategory: result.inferredCategory,
    });
  });

  const summary = {
    totalRows: rows.length,
    acceptedRows: acceptedRows.length,
    skippedRows: skippedRows.length,
    inferredCategories: inferredCategoryCount,
  };

  return savePreview('transactions', {
    acceptedRows,
    skippedRows,
    summary,
  });
}

function buildMlDatasetPreview(rows = []) {
  const acceptedRows = [];
  const skippedRows = [];

  rows.forEach((row, index) => {
    const result = inspectMlDatasetRow(row, index + 1);
    if (!result.accepted) {
      skippedRows.push({
        rowNumber: index + 2,
        reason: result.reason,
        reasons: result.reasons,
        raw: row,
      });
      return;
    }

    acceptedRows.push({
      rowNumber: index + 2,
      raw: row,
      normalized: result.normalized,
    });
  });

  const summary = {
    totalRows: rows.length,
    acceptedRows: acceptedRows.length,
    skippedRows: skippedRows.length,
    requiredColumns: REQUIRED_COLUMNS,
  };

  return savePreview('ml_dataset', {
    acceptedRows,
    skippedRows,
    summary,
  });
}

function getPreview(previewId) {
  cleanupExpiredPreviews();
  const preview = previewStore.get(previewId);
  if (!preview) {
    return null;
  }
  return preview;
}

function consumePreview(previewId) {
  const preview = getPreview(previewId);
  if (!preview) {
    return null;
  }

  previewStore.delete(previewId);
  return preview;
}

module.exports = {
  buildImportPreview,
  buildMlDatasetPreview,
  getPreview,
  consumePreview,
};
