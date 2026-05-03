import json

CLASSIFICATION_LABELS = {
    'knn_euclidean':           'kNN (Dist. Euclidiana)',
    'knn_manhattan':           'kNN (Dist. Manhattan)',
    'naive_bayes_univariado':  'Bayesiano (Univariado)',
    'naive_bayes_multivariado':'Bayesiano (Multivariado)',
}

REGRESSION_LABELS = {
    'knn_euclidean':    'kNN (Dist. Euclidiana)',
    'knn_manhattan':    'kNN (Dist. Manhattan)',
    'linear_regression':'Regressão Linear Múltipla',
}

CLASSIFICATION_COLS = [
    ('accuracy',  'Acurácia'),
    ('precision', 'Precisão'),
    ('recall',    'Recall'),
    ('f1_score',  'F1-Score'),
]

REGRESSION_COLS = [
    ('r2_score',          'R²'),
    ('r2_score_adjusted', 'R² Ajustado'),
]

TIME_COLS = [
    ('time_fit',  'T. Treino (s)'),
    ('time_pred', 'T. Teste (s)'),
]


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def print_table(json_path: str, model_labels: dict, metric_cols: list):
    with open(json_path) as f:
        data = json.load(f)

    all_cols = metric_cols + TIME_COLS

    # Larguras de coluna
    col_model_w = max(len(label) for label in model_labels.values()) + 2
    col_w = max(len(header) for _, header in all_cols) + 4
    col_w = max(col_w, len("0.0000 ± 0.0000") + 4)

    header_parts = [f"{'Modelo':<{col_model_w}}"]
    for _, header in all_cols:
        header_parts.append(f"{header:^{col_w}}")
    header_line = "| " + " | ".join(header_parts) + " |"

    separator = "+" + "+".join(["-" * (col_model_w + 2)] + ["-" * (col_w + 2)] * len(all_cols)) + "+"

    print(separator)
    print(header_line)
    print(separator)

    for model_key, label in model_labels.items():
        if model_key not in data:
            continue
        row = data[model_key]
        cells = [f"{label:<{col_model_w}}"]
        for metric_key, _ in metric_cols:
            mean = row.get(f'mean-{metric_key}', 0)
            std  = row.get(f'std-{metric_key}', 0)
            cells.append(f"{fmt(mean, std):^{col_w}}")
        for time_key, _ in TIME_COLS:
            mean = row.get(f'mean-{time_key}', 0)
            std  = row.get(f'std-{time_key}', 0)
            cells.append(f"{fmt(mean, std):^{col_w}}")
        print("| " + " | ".join(cells) + " |")

    print(separator)


if __name__ == '__main__':
    print("\n=== CLASSIFICAÇÃO ===\n")
    print_table(
        'classification_table.json',
        CLASSIFICATION_LABELS,
        CLASSIFICATION_COLS,
    )

    print("\n=== REGRESSÃO ===\n")
    print_table(
        'regression_table.json',
        REGRESSION_LABELS,
        REGRESSION_COLS,
    )