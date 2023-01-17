# Calculator black and scholes(greeks for options)
# Calculadora de black and scholes

## Introdução
O modelo de black and scholes foi desenvolvido na década de 70 pelos economistas Fischer Black, Robert Merton e Myron Scholes. Ele é amplamente utilizado para calcular os valores teóricos de opções no mercado de derivativos.

### O que é um derivativo?


### O modelo
#### Entradas
O modelo de black and scholes possui 5 entradas:
- Preço da opção*: Preço de mercado da opção
- Preço do ativo[S]: Valor do underlying para aquele derivativo
- Strike da opção[K]: Valor de strike do underlying
- Tempo(dte): Quantidades de dias restantes para a opção expirar
- Taxa de juros[r]: Taxa de juros utilizada como referência
- Volatilidade[vol]*: Volatilidade que o mercado está prevendo para o vencimento da opção

\* Pode ser encontrado utilizando a calculadora, caso as outras variáveis sejam conhecidas

#### Distribuição normal
O black considera que a variação do underlying segue uma distribuição normal acumulativo, utilizando como parâmetro um desvio padrão. Desse modo, como pode ser observado na imagem abaixo, o modelo utilizado infere que há aproximadamente 68% de probabilidade do underlying na data de expiração estar dentro do intervalo de 1 desvio padrão

#### Fórmula
O modelo de black and scholes possui as seguintes fórmulas

- Para Call:
    $$ C(S, K, r, dte) = N(d_1) \cdot S - N(d_2) \cdot Ke^{-r \cdot dte} $$
- Para Put:
    - $ P(S, K, r, dte) = N(-d_2) \cdot Ke^{-r \cdot dte} - N(-d_1) \cdot S $
- Onde:
    - S: underlying da opção
    - K: Strike da opção
    - r: Taxa de juros escolhida
    - dte: Tempo restante para expirar a opção
    - N: Função da distribuição normal

#### d1 e d2
Estas são as variaveis utilizadas para determinar a distribuição normal da opção, podendo ser encontradas das seguinte forma

$ d_1 = \frac{log(\frac{S}{K}) + (r + 0,5 \cdot vol^2)dte}{vol \cdot \sqrt{dte}}$

$ d_2 = \frac{log(\frac{S}{K}) + (r - 0,5 \cdot vol^2)dte}{vol \cdot \sqrt{dte}}$

OBS: Para encontrar $d_2$ basta inverter o sinal de $0,5 \cdot vol^2$ em $d_1$

## Referências
- https://www.suno.com.br/artigos/black-scholes/
- https://capitalresearch.com.br/blog/black-and-scholes/
