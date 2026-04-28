#!/bin/bash

# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install -r requeriments.txt

echo "Ambiente virtual criado e dependências instaladas com sucesso!"