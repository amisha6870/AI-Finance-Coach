import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = Path(os.environ.get('ML_DATASET_PATH', ROOT_DIR / 'dataset' / 'financial_behavior.csv'))
DEFAULT_ARTIFACT_DIR = Path(os.environ.get('ML_ARTIFACT_DIR', ROOT_DIR / 'models'))
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    'income',
    'total_expenses',
    'food_expense',
    'shopping_expense',
    'rent_expense',
    'savings',
    'remaining_balance',
    'overspending_label',
]

NUMERIC_COLUMNS = [
    'income',
    'total_expenses',
    'food_expense',
    'shopping_expense',
    'rent_expense',
    'savings',
    'remaining_balance',
    'month_index',
]

LOGISTIC_FEATURES = [
    'income',
    'total_expenses',
    'food_expense',
    'shopping_expense',
    'rent_expense',
    'savings',
    'remaining_balance',
    'savings_rate',
    'spend_ratio',
    'discretionary_spending_percentage',
]

KMEANS_FEATURES = [
    'savings_rate',
    'spend_ratio',
    'discretionary_spending_percentage',
]

LINEAR_FEATURES = [
    'month_index',
    'income',
    'food_expense',
    'shopping_expense',
    'rent_expense',
    'savings',
    'remaining_balance',
    'savings_rate',
    'spend_ratio',
    'discretionary_spending_percentage',
]

ARTIFACT_FILES = {
    'logistic': 'logistic_model.pkl',
    'kmeans': 'kmeans_model.pkl',
    'linear': 'linear_model.pkl',
    'metadata': 'training_metadata.json',
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_artifact_dir(artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)


def normalize_input_row(row: Dict) -> Dict:
    category_totals = row.get('categoryTotals') or row.get('category_totals') or {}
    income = _safe_float(row.get('income'))
    total_expenses = _safe_float(row.get('total_expenses', row.get('totalExpenses', row.get('expenses'))))
    food_expense = _safe_float(row.get('food_expense', row.get('foodExpense', category_totals.get('Food & Dining'))))
    shopping_expense = _safe_float(row.get('shopping_expense', row.get('shoppingExpense', category_totals.get('Shopping'))))
    rent_expense = _safe_float(
        row.get(
            'rent_expense',
            row.get('rentExpense', category_totals.get('Bills & Utilities', category_totals.get('Rent'))),
        )
    )
    savings = _safe_float(row.get('savings', income - total_expenses))
    remaining_balance = _safe_float(row.get('remaining_balance', row.get('remainingBalance')))
    month_index = int(_safe_float(row.get('month_index', row.get('monthIndex', 1)), 1))

    result = {
        'income': income,
        'total_expenses': total_expenses,
        'food_expense': max(0.0, food_expense),
        'shopping_expense': max(0.0, shopping_expense),
        'rent_expense': max(0.0, rent_expense),
        'savings': savings,
        'remaining_balance': remaining_balance,
        'month_index': max(1, month_index),
    }

    result['savings_rate'] = savings / income if income > 0 else 0.0
    result['spend_ratio'] = total_expenses / income if income > 0 else (1.0 if total_expenses > 0 else 0.0)
    result['discretionary_spending_percentage'] = (
        (result['food_expense'] + result['shopping_expense']) / total_expenses if total_expenses > 0 else 0.0
    )

    return result


def validate_and_prepare_dataframe(dataset_path: Path) -> Tuple[pd.DataFrame, Dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found at {dataset_path}')

    dataframe = pd.read_csv(dataset_path)
    original_rows = len(dataframe)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f'Missing required dataset columns: {", ".join(missing_columns)}')

    if 'month_index' not in dataframe.columns:
        dataframe['month_index'] = range(1, len(dataframe) + 1)

    for column in NUMERIC_COLUMNS + ['overspending_label']:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

    invalid_label_mask = ~dataframe['overspending_label'].isin([0, 1])
    invalid_label_rows = int(invalid_label_mask.sum())
    dataframe = dataframe.loc[~invalid_label_mask].copy()

    numeric_fill_columns = [column for column in NUMERIC_COLUMNS if column in dataframe.columns]
    medians = {}
    for column in numeric_fill_columns:
        median_value = dataframe[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        dataframe[column] = dataframe[column].fillna(median_value)
        medians[column] = float(median_value)

    dataframe['savings_rate'] = dataframe.apply(
        lambda row: (row['savings'] / row['income']) if row['income'] > 0 else 0.0,
        axis=1,
    )
    dataframe['spend_ratio'] = dataframe.apply(
        lambda row: (row['total_expenses'] / row['income']) if row['income'] > 0 else (1.0 if row['total_expenses'] > 0 else 0.0),
        axis=1,
    )
    dataframe['discretionary_spending_percentage'] = dataframe.apply(
        lambda row: ((row['food_expense'] + row['shopping_expense']) / row['total_expenses']) if row['total_expenses'] > 0 else 0.0,
        axis=1,
    )

    for column in LOGISTIC_FEATURES + KMEANS_FEATURES + LINEAR_FEATURES:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype(float)

    stats = {
        'row_count': int(len(dataframe)),
        'original_row_count': int(original_rows),
        'invalid_rows_dropped': int(original_rows - len(dataframe)),
        'invalid_label_rows_dropped': invalid_label_rows,
        'medians': medians,
        'dataset_path': str(dataset_path),
    }

    if len(dataframe) < 10:
        raise ValueError('Dataset needs at least 10 valid rows for stable training')

    if dataframe['overspending_label'].nunique() < 2:
        raise ValueError('Dataset needs both overspending classes (0 and 1) for logistic regression training')

    return dataframe, stats


def build_logistic_artifact(dataframe: pd.DataFrame):
    x = dataframe[LOGISTIC_FEATURES]
    y = dataframe['overspending_label'].astype(int)

    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE)),
    ])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    confusion = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    metrics = {
        'accuracy': round(float(accuracy_score(y_test, predictions)), 4),
        'precision': round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        'test_samples': int(len(x_test)),
        'train_samples': int(len(x_train)),
        'confusion_matrix': confusion,
        'average_probability': round(float(probabilities.mean()) if len(probabilities) else 0.0, 4),
    }

    return {
        'pipeline': pipeline,
        'features': LOGISTIC_FEATURES,
        'metrics': metrics,
        'model_name': 'Logistic Regression',
    }


def build_kmeans_artifact(dataframe: pd.DataFrame):
    cluster_frame = dataframe[KMEANS_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_frame)
    model = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    model.fit(scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)
    score_rows = []
    for index, center in enumerate(centers):
        savings_rate, spend_ratio, discretionary_share = center.tolist()
        score_rows.append({
            'cluster_id': int(index),
            'score': float(spend_ratio + discretionary_share - savings_rate),
            'center': {
                'savings_rate': round(float(savings_rate), 4),
                'spend_ratio': round(float(spend_ratio), 4),
                'discretionary_spending_percentage': round(float(discretionary_share), 4),
            },
        })

    ordered = sorted(score_rows, key=lambda item: item['score'])
    label_map = {
        ordered[0]['cluster_id']: 'Low Spender',
        ordered[1]['cluster_id']: 'Moderate Spender',
        ordered[2]['cluster_id']: 'High Spender',
    }

    silhouette = 0.0
    if len(cluster_frame) > 3:
        try:
            silhouette = float(silhouette_score(scaled, model.labels_))
        except Exception:
            silhouette = 0.0

    cluster_counts = dataframe.assign(cluster_id=model.labels_).groupby('cluster_id').size().to_dict()

    metrics = {
        'inertia': round(float(model.inertia_), 4),
        'silhouette_score': round(silhouette, 4),
        'cluster_label_map': {str(key): value for key, value in label_map.items()},
        'cluster_centers': score_rows,
        'cluster_distribution': {label_map.get(cluster_id, str(cluster_id)): int(count) for cluster_id, count in cluster_counts.items()},
        'samples': int(len(cluster_frame)),
    }

    return {
        'model': model,
        'scaler': scaler,
        'features': KMEANS_FEATURES,
        'label_map': label_map,
        'metrics': metrics,
        'model_name': 'K-Means',
    }


def build_linear_artifact(dataframe: pd.DataFrame):
    x = dataframe[LINEAR_FEATURES]
    y = dataframe['total_expenses'].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    r2 = r2_score(y_test, predictions) if len(x_test) > 1 else 0.0
    mae = mean_absolute_error(y_test, predictions) if len(x_test) > 0 else 0.0

    metrics = {
        'r2_score': round(float(r2), 4),
        'mean_absolute_error': round(float(mae), 4),
        'test_samples': int(len(x_test)),
        'train_samples': int(len(x_train)),
    }

    return {
        'pipeline': pipeline,
        'features': LINEAR_FEATURES,
        'metrics': metrics,
        'model_name': 'Linear Regression',
    }


def save_training_artifacts(logistic_artifact, kmeans_artifact, linear_artifact, dataset_stats: Dict, artifact_dir: Path, dataset_path: Path):
    ensure_artifact_dir(artifact_dir)

    joblib.dump(logistic_artifact, artifact_dir / ARTIFACT_FILES['logistic'])
    joblib.dump(kmeans_artifact, artifact_dir / ARTIFACT_FILES['kmeans'])
    joblib.dump(linear_artifact, artifact_dir / ARTIFACT_FILES['linear'])

    metadata = {
        'trained_at': utc_now_iso(),
        'dataset_path': str(dataset_path),
        'dataset_stats': dataset_stats,
        'artifacts': ARTIFACT_FILES,
        'models': {
            'logistic_regression': logistic_artifact['metrics'],
            'kmeans': kmeans_artifact['metrics'],
            'linear_regression': linear_artifact['metrics'],
        },
    }

    (artifact_dir / ARTIFACT_FILES['metadata']).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def train_models(dataset_path: Path = DEFAULT_DATASET_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    dataframe, dataset_stats = validate_and_prepare_dataframe(dataset_path)
    logistic_artifact = build_logistic_artifact(dataframe)
    kmeans_artifact = build_kmeans_artifact(dataframe)
    linear_artifact = build_linear_artifact(dataframe)
    metadata = save_training_artifacts(
        logistic_artifact,
        kmeans_artifact,
        linear_artifact,
        dataset_stats,
        artifact_dir,
        dataset_path,
    )

    return {
        'status': 'trained',
        'trained_at': metadata['trained_at'],
        'dataset': dataset_stats,
        'models': {
            'logistic_regression': logistic_artifact['metrics'],
            'kmeans': kmeans_artifact['metrics'],
            'linear_regression': linear_artifact['metrics'],
        },
        'artifact_dir': str(artifact_dir),
    }


def artifact_paths(artifact_dir: Path):
    return {key: artifact_dir / filename for key, filename in ARTIFACT_FILES.items()}


def artifacts_ready(artifact_dir: Path) -> bool:
    paths = artifact_paths(artifact_dir)
    return all(path.exists() for path in [paths['logistic'], paths['kmeans'], paths['linear'], paths['metadata']])


def load_metadata(artifact_dir: Path) -> Dict:
    metadata_path = artifact_paths(artifact_dir)['metadata']
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def ensure_trained(dataset_path: Path = DEFAULT_DATASET_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    if not artifacts_ready(artifact_dir):
        train_models(dataset_path=dataset_path, artifact_dir=artifact_dir)


def load_artifacts(dataset_path: Path = DEFAULT_DATASET_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    ensure_trained(dataset_path=dataset_path, artifact_dir=artifact_dir)
    paths = artifact_paths(artifact_dir)

    return {
        'logistic': joblib.load(paths['logistic']),
        'kmeans': joblib.load(paths['kmeans']),
        'linear': joblib.load(paths['linear']),
        'metadata': load_metadata(artifact_dir),
    }


def build_prediction_payload(raw_payload: Dict) -> Dict:
    monthly_history = raw_payload.get('monthlyHistory') or raw_payload.get('rows') or []
    current = raw_payload.get('current') or {}
    current_row = normalize_input_row(current)

    if monthly_history:
        month_index = max(int(_safe_float(row.get('month_index', row.get('monthIndex', index + 1)), index + 1)) for index, row in enumerate(monthly_history)) + 1
        current_row['month_index'] = month_index

    return current_row


def predict(payload: Dict, dataset_path: Path = DEFAULT_DATASET_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    bundles = load_artifacts(dataset_path=dataset_path, artifact_dir=artifact_dir)
    current_row = build_prediction_payload(payload)

    logistic_bundle = bundles['logistic']
    logistic_frame = pd.DataFrame([[current_row[feature] for feature in logistic_bundle['features']]], columns=logistic_bundle['features'])
    probability = float(logistic_bundle['pipeline'].predict_proba(logistic_frame)[0][1])

    if probability >= 0.7:
        risk_level = 'High'
        risk_text = 'High overspending risk'
    elif probability >= 0.4:
        risk_level = 'Medium'
        risk_text = 'Moderate overspending risk'
    else:
        risk_level = 'Low'
        risk_text = 'Low overspending risk'

    confidence_percent = int(round(max(probability, 1 - probability) * 100))

    kmeans_bundle = bundles['kmeans']
    cluster_frame = pd.DataFrame([[current_row[feature] for feature in kmeans_bundle['features']]], columns=kmeans_bundle['features'])
    scaled_cluster = kmeans_bundle['scaler'].transform(cluster_frame)
    cluster_id = int(kmeans_bundle['model'].predict(scaled_cluster)[0])
    spender_type = kmeans_bundle['label_map'].get(cluster_id, 'Moderate Spender')

    linear_bundle = bundles['linear']
    linear_frame = pd.DataFrame([[current_row[feature] for feature in linear_bundle['features']]], columns=linear_bundle['features'])
    predicted_expense = max(0.0, float(linear_bundle['pipeline'].predict(linear_frame)[0]))
    current_expenses = current_row['total_expenses']

    if current_expenses <= 0:
        direction = 'stable'
    else:
        variance_ratio = (predicted_expense - current_expenses) / current_expenses
        if variance_ratio > 0.08:
            direction = 'rising'
        elif variance_ratio < -0.08:
            direction = 'falling'
        else:
            direction = 'stable'

    metadata = bundles['metadata'] or {}
    logistic_metrics = logistic_bundle.get('metrics', {})
    linear_metrics = linear_bundle.get('metrics', {})
    kmeans_metrics = kmeans_bundle.get('metrics', {})
    trend_confidence = max(0.0, min(100.0, float(linear_metrics.get('r2_score', 0.0)) * 100.0))

    return {
        'source': 'trained_model',
        'overspending_risk': risk_level,
        'confidence': confidence_percent,
        'predicted_expense': round(predicted_expense, 2),
        'spender_type': spender_type,
        'overspending': {
            'prediction': risk_text,
            'probability': round(probability, 4),
            'confidence': confidence_percent,
            'riskLevel': risk_level.lower(),
            'model': logistic_bundle.get('model_name', 'Logistic Regression'),
            'metrics': logistic_metrics,
        },
        'behavior': {
            'segment': spender_type,
            'cluster': cluster_id,
            'model': kmeans_bundle.get('model_name', 'K-Means'),
            'metrics': kmeans_metrics,
        },
        'trend': {
            'nextMonthExpense': round(predicted_expense, 2),
            'direction': direction,
            'confidenceScore': round(trend_confidence, 2),
            'model': linear_bundle.get('model_name', 'Linear Regression'),
            'metrics': linear_metrics,
        },
        'training': {
            'status': 'ready',
            'datasetPath': str(metadata.get('dataset_path', dataset_path)),
            'trainedAt': metadata.get('trained_at'),
            'sampleCount': int(metadata.get('dataset_stats', {}).get('row_count', 0)),
            'invalidRowsDropped': int(metadata.get('dataset_stats', {}).get('invalid_rows_dropped', 0)),
            'artifactDir': str(artifact_dir),
        },
    }


def runtime_status(dataset_path: Path = DEFAULT_DATASET_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    dataset_exists = dataset_path.exists()
    ready = dataset_exists and artifacts_ready(artifact_dir)
    metadata = load_metadata(artifact_dir) if ready else {}

    return {
        'ready': ready,
        'pythonCommand': sys.executable,
        'scriptPath': str(Path(__file__).resolve()),
        'sklearn': True,
        'pandas': True,
        'joblib': True,
        'datasetExists': dataset_exists,
        'datasetPath': str(dataset_path),
        'artifactDir': str(artifact_dir),
        'artifactsReady': artifacts_ready(artifact_dir),
        'trainedAt': metadata.get('trained_at'),
        'error': None if dataset_exists else f'Dataset not found at {dataset_path}',
    }


def parse_args():
    parser = argparse.ArgumentParser(description='MountDash ML training and prediction pipeline')
    parser.add_argument('command', nargs='?', default='predict', choices=['train', 'predict', 'runtime'])
    parser.add_argument('--dataset', dest='dataset_path', default=str(DEFAULT_DATASET_PATH))
    parser.add_argument('--artifacts', dest='artifact_dir', default=str(DEFAULT_ARTIFACT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    artifact_dir = Path(args.artifact_dir)

    try:
        if args.command == 'train':
            result = train_models(dataset_path=dataset_path, artifact_dir=artifact_dir)
        elif args.command == 'runtime':
            result = runtime_status(dataset_path=dataset_path, artifact_dir=artifact_dir)
        else:
            raw = sys.stdin.read()
            payload = json.loads(raw or '{}')
            result = predict(payload, dataset_path=dataset_path, artifact_dir=artifact_dir)

        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({'error': str(error), 'command': args.command}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()



