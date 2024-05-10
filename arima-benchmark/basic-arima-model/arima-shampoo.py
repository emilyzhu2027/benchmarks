# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://github.com/alkaline-ml/pmdarima/blob/master/examples/quick_start_example.ipynb

# -------- Using Statsmodels ---------------------------------------------------------
# fit an ARIMA model and plot residual errors
# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
def parser(x):
 return datetime.datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo.csv', header=0, index_col=0, parse_dates=True, date_parser=parser)
series = series.squeeze()
series.index = series.index.to_period('M')
# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')


# -------- Using AutoArima ---------------------------------------------------------
from pmdarima.arima import ARIMA
import pmdarima as pm
import numpy as np
def calcsmape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast)))

stepwise_fit = pm.auto_arima(train, start_p=1, start_q=1, test = 'adf', max_p=10, max_q=10, m=1,
                             start_P=0, seasonal=False, d=1, D=None, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise

print(stepwise_fit.summary())
prediction = stepwise_fit.predict(n_periods=len(test))
smape=calcsmape(test, prediction)
print(smape)