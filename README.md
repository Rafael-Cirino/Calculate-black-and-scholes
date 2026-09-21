# Black-Scholes Analytics

## English version

A small Python project for Black-Scholes option pricing and Greek calculations, organized for `uv`, `Polars`, and `Typer`.

### Overview

This project keeps the original Black-Scholes numerical logic but restructures it into a clean package so it can be run as a proper CLI tool. It is useful for quick option analysis and for processing option datasets in CSV form.

### Requirements

- Python 3.12+
- uv

### Quick start

From the project root:

```bash
uv sync
uv run black-scholes --help
```

### Commands

#### Example pricing

```bash
uv run black-scholes example
```

#### Direct price calculation

```bash
uv run black-scholes price --s 100 --k 100 --t 1 --r 0.05 --q 0 --volatility 0.2 --option call
```

#### Analyze a CSV dataset

```bash
uv run black-scholes analyze ./btc_dte_early_stop_exp_0.2_all-time_24h.csv --option C
```

### Project structure

```text
.
├── README.md
├── README.pt-BR.md
├── pyproject.toml
├── src/
│   └── black_scholes/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       └── pricing.py
└── tests/
    └── test_pricing.py
```

### Example usage in Python

```python
from black_scholes.pricing import Greeks, option_price

price = option_price(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    q=0.0,
    volatility=0.2,
    option="call",
)

greeks = Greeks(S=100.0, K=100.0, T=1.0, r=0.05, volatility=0.2, option="call")
print(price)
print(greeks.calculate_all())
```

### Notes

The original project was a single script. This version organizes the logic into reusable modules, improves the developer workflow, and keeps the pricing logic compatible with the previous implementation.

----------------------------------------------------------------------------------------------------------------------------------

## Português version

Um pequeno projeto em Python para precificação de opções pelo modelo Black-Scholes e cálculo de gregas, organizado para `uv`, `Polars` e `Typer`.

### Visão geral

Este projeto preserva a lógica numérica original do Black-Scholes, mas reestrutura o código em um pacote mais limpo para funcionar como ferramenta de linha de comando. Ele é útil para análise rápida de opções e processamento de arquivos CSV de dados de mercado.

### Requisitos

- Python 3.12+
- uv

### Início rápido

Na raiz do projeto:

```bash
uv sync
uv run black-scholes --help
```

### Comandos

#### Exemplo de precificação

```bash
uv run black-scholes example
```

#### Cálculo direto de preço

```bash
uv run black-scholes price --s 100 --k 100 --t 1 --r 0.05 --q 0 --volatility 0.2 --option call
```

#### Analisar um arquivo CSV

```bash
uv run black-scholes analyze ./btc_dte_early_stop_exp_0.2_all-time_24h.csv --option C
```

### Estrutura do projeto

```text
.
├── README.md
├── README.pt-BR.md
├── pyproject.toml
├── src/
│   └── black_scholes/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       └── pricing.py
└── tests/
    └── test_pricing.py
```

### Uso em Python

```python
from black_scholes.pricing import Greeks, option_price

price = option_price(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    q=0.0,
    volatility=0.2,
    option="call",
)

greeks = Greeks(S=100.0, K=100.0, T=1.0, r=0.05, volatility=0.2, option="call")
print(price)
print(greeks.calculate_all())
```

### Observações

O projeto original era um único script. Esta versão organiza a lógica em módulos reutilizáveis, melhora o fluxo de desenvolvimento e mantém a precificação compatível com a implementação anterior.
