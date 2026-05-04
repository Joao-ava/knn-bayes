from train import train_classification_model, train_regressor_model
from table import print_table, CLASSIFICATION_LABELS, CLASSIFICATION_COLS, REGRESSION_LABELS, REGRESSION_COLS

if __name__ == '__main__':
    k = 5
    train_classification_model(k)
    train_regressor_model(k)
    
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