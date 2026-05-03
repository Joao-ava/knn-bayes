from train import train_classification_model, train_regressor_model

if __name__ == '__main__':
    k = 5
    train_classification_model(k)
    train_regressor_model(k)