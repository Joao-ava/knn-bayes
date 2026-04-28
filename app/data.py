import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from data_preprocessing import DataPreprocessing

"""
docstring:
Módulo para carregamento de dados, incluindo funções específicas para cada dataset.

Atributos:
- CURRENT_PATH: caminho do diretório atual (app/)
- DATA_PATH: caminho do diretório de dados (app/data/)

Funções:
- load_santander: carrega o dataset de transações Santander a partir de um arquivo Parquet.
- load_fraud_detection: carrega o dataset de detecção de fraude a partir de um arquivo Parquet.
cada função retorna os dados em formato de arrays numpy, prontos para pré-processamento e modelagem.

Uso:
1. importar as funções de carregamento de dados.
2. chamar a função desejada para obter os dados X e y.

Exemplo:
from data import load_santander, load_fraud_detection
X_santander, y_santander = load_santander()
X_fraud, y_fraud = load_fraud_detection()


"""

CURRENT_PATH = Path(__file__).parent
DATA_PATH = CURRENT_PATH.parent / 'data'


def load_santander():
    """
    Carrega os dados de transações Santander.
    Parquet: Santander Transaction dataset
    Retorno: X (features) com shape (n, n_features), y (target) com shape (n,)
    """
    filepath = DATA_PATH / 'santander_transaction.pq'
    
    # Ler o arquivo Parquet usando pyarrow
    table = pq.read_table(filepath)
    
    # Converter para numpy arrays diretamente
    n_cols = table.num_columns
    X = table.to_pydict()
    
    # Converter dicionário para matriz numpy
    # Assumindo que a última coluna é o target
    columns = list(X.keys())
    y = np.array(X[columns[-1]])
    X = np.array([X[col] for col in columns[:-1]]).T
    
    return X, y


def load_fraud_detection():
    """
    Carrega os dados de detecção de fraude.
    Parquet: Fraud Detection dataset
    Retorno: X (features) com shape (n, n_features), y (target) com shape (n,)
    """
    filepath = DATA_PATH / 'fraud_detection.pq'
    
    # Ler o arquivo Parquet usando pyarrow
    table = pq.read_table(filepath)
    
    # Converter para numpy arrays diretamente
    X = table.to_pydict()
    
    # Converter dicionário para matriz numpy
    # Assumindo que a última coluna é o target
    columns = list(X.keys())
    y = np.array(X[columns[-1]])
    X = np.array([X[col] for col in columns[:-1]]).T
    
    return X, y

