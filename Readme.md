# Trading algos

## Introduction
This repo contains backtesting code for various trading strategies. I mainly focus on Pairs Trading so you'll which essentially focuses on finding cointegrated stocks and then applying various permutations of the Pairs Trading strategy.

## Pair trading
In a few words, if a pair of stocks have behaved similarly in the past we can expect that even if one of them deviates from the other it will eventually normalize. Given this assumption we can execute a trade while keeping a hedge on the other stock.

This is a market neutral strategy and will work in any market condition.


## Setup

>`conda env create -f conda-env.yml`

>`conda activate trading-algos`

>`conda activate trading-algos`

>`pip install -r requirements.txt`

>`jupyter notebook`

## Datasource
I used EOD https://eodhistoricaldata.com which requires an API key.


enjoy :)