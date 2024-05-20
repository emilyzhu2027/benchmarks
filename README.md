### ARIMA Benchmark
1. run arima-benchmark/aerobench/arima_benchmark_f16gcas.py (ignore arima_benchmark.py)
- this will run 2000 trials of the F16 flight simulator, where the initial parameters are randomized, and use an ARIMA model to predict the altitude of the aircraft. for each trial, it will output some plots and evaluate each model based on the RMSE. the RMSEs of each trial will then be saved in arima-benchmark/aerobench/arima_results.csv.


### LSTM Benchmark
1. run arima-benchmark/aerobench/lstm_benchmark.py
- run one trial of F16 flight simulator, and uses LSTM model to predict altitude of aircraft. although it technically runs, it's not quite accurate however and i'm working to fix that!
