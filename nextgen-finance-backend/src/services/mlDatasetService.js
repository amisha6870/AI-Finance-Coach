const fs = require('fs');
const path = require('path');

const REQUIRED_COLUMNS = [
  'income',
  'total_expenses',
  'food_expense',
  'shopping_expense',
  'rent_expense',
  'savings',
  'remaining_balance',
  'overspending_label',
];

const OPTIONAL_COLUMNS = ['month_index'];
const OUTPUT_COLUMNS = [...REQUIRED_COLUMNS, ...OPTIONAL_COLUMNS];

function stripBom(value = '') {
  return String(value || '').replace(/^\uFEFF/, '');
}

function normalizeHeaderName(header = '') {
  return stripBom(header)
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_');
}

function buildHeaderMap(row = {}) {
  return Object.keys(row).reduce((accumulator, key) => {
    accumulator[normalizeHeaderName(key)] = row[key];
    return accumulator;
  }, {});
}

function parseNumericValue(value) {
  if (value === null || value === undefined) return { ok: false, value: null };
  const normalized = String(value).trim();
  if (!normalized) return { ok: false, value: null };
  const cleaned = normalized.replace(/,/g, '').replace(/[^0-9.-]/g, '');
  if (!cleaned || cleaned === '-' || cleaned === '.' || cleaned === '-.') return { ok: false, value: null };
  const parsed = Number(cleaned);
  if (!Number.isFinite(parsed)) return { ok: false, value: null };
  return { ok: true, value: parsed };
}

function parseBinaryLabel(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === '0' || normalized === '0.0' || normalized === 'false' || normalized === 'no') {
    return { ok: true, value: 0 };
  }
  if (normalized === '1' || normalized === '1.0' || normalized === 'true' || normalized === 'yes') {
    return { ok: true, value: 1 };
  }
  return { ok: false, value: null };
}

function detectMlDatasetSchema(rows = []) {
  const firstRow = rows[0] || {};
  const headerMap = buildHeaderMap(firstRow);
  return REQUIRED_COLUMNS.every((column) => Object.prototype.hasOwnProperty.call(headerMap, column));
}

function inspectMlDatasetRow(row = {}, rowNumber = 0) {
  const headerMap = buildHeaderMap(row);
  const reasons = [];
  const normalized = {};

  for (const column of REQUIRED_COLUMNS) {
    if (!Object.prototype.hasOwnProperty.call(headerMap, column)) {
      reasons.push(`Missing required column: ${column}`);
      continue;
    }

    if (column === 'overspending_label') {
      const parsedLabel = parseBinaryLabel(headerMap[column]);
      if (!parsedLabel.ok) {
        reasons.push('overspending_label must be 0 or 1');
      } else {
        normalized[column] = parsedLabel.value;
      }
      continue;
    }

    const parsedNumber = parseNumericValue(headerMap[column]);
    if (!parsedNumber.ok) {
      reasons.push(`${column} must be a valid number`);
    } else {
      normalized[column] = parsedNumber.value;
    }
  }

  if (Object.prototype.hasOwnProperty.call(headerMap, 'month_index')) {
    const parsedMonthIndex = parseNumericValue(headerMap.month_index);
    if (!parsedMonthIndex.ok || parsedMonthIndex.value <= 0) {
      reasons.push('month_index must be a positive number when provided');
    } else {
      normalized.month_index = Math.round(parsedMonthIndex.value);
    }
  } else {
    normalized.month_index = rowNumber;
  }

  if (reasons.length > 0) {
    return {
      accepted: false,
      reasons,
      reason: reasons.join('; '),
    };
  }

  return {
    accepted: true,
    normalized,
  };
}

function serializeMlDatasetRows(rows = []) {
  const headerLine = OUTPUT_COLUMNS.join(',');
  const dataLines = rows.map((row) => OUTPUT_COLUMNS.map((column) => row[column] ?? '').join(','));
  return [headerLine, ...dataLines].join('\n');
}

function persistMlDataset(rows = [], datasetPath) {
  const nextPath = datasetPath || path.join(__dirname, '..', '..', 'dataset', 'financial_behavior.csv');
  fs.mkdirSync(path.dirname(nextPath), { recursive: true });
  fs.writeFileSync(nextPath, serializeMlDatasetRows(rows), 'utf-8');
  return nextPath;
}

module.exports = {
  REQUIRED_COLUMNS,
  OPTIONAL_COLUMNS,
  OUTPUT_COLUMNS,
  normalizeHeaderName,
  detectMlDatasetSchema,
  inspectMlDatasetRow,
  serializeMlDatasetRows,
  persistMlDataset,
};

