# EXP-001: Orderbook Imbalance

## Гипотеза

L1 order book imbalance (bid_sz - ask_sz) / (bid_sz + ask_sz) предсказывает краткосрочное движение mid price.

## План

1. Загрузить L10 orderbook snapshot для BTC-USDT-SWAP.
2. Вычислить L1 imbalance.
3. Построить label: future mid move (horizon 10 rows).
4. Оценить hit rate, precision long/short для простого imbalance-сигнала.
