# Test and evaluate ARIMA model on 2000 different trials of F16 flight simulator
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_GCAS import main_run_gcas
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pmdarima.arima.stationarity import ADFTest
import re


def main():
    results_df = pd.DataFrame(columns = ['trial_num', 'data_type', 'p', 'd', 'q', 'rmse'])
    for i in range(20): # 20 trials - can change 
        df = main_run_gcas() # Run flight simulator which uses randomized initial parameters
    
        adf_test = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test.should_diff(df['alt'])

        if(should_diff): 
            data_type = 'not stationary'
        else:
            data_type = 'stationary'

        
        train_size = int(0.7*len(df)) # Split train and test
        train = df.iloc[0:train_size]
        test = df.iloc[train_size:]


        # ----------------- ARIMA ---------------------
        model = auto_arima(train['alt'], start_p=1, start_q=1, # Find best parameters for ARIMA model
                        test='adf',
                        max_p=10, max_q=10,
                        m=1,  
                        d=3,                    
                        seasonal=False,   
                        start_P=0, 
                        D=None, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
        summary_string = str(model.summary())
        param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)',summary_string)
        p,d,q = int(param[0][0]) , int(param[0][1]) , int(param[0][2])
        prediction, confint = model.predict(len(test), return_conf_int=True) # Make predictions
        cf= pd.DataFrame(confint)
        mse = mean_squared_error(test['alt'], prediction)
        rmse = mse**0.5

        results_df = pd.concat([pd.DataFrame([[i+1, data_type, p, d, q, rmse]], columns=results_df.columns), results_df], ignore_index=True)

        """
        # Uncomment this part to plot prediction against test data
        plt.figure(figsize=(14,7))
        plt.plot(train['alt'], label='Training Data')
        plt.plot(test['alt'], label='Actual Data', color='orange')
        prediction_df = pd.DataFrame(prediction)
        prediction_df.index = test.index
        plt.plot(prediction_df, label='Forecasted Data', color='green')
        plt.fill_between(test.index, 
                        cf[0], 
                        cf[1], 
                        color='k', alpha=.15)
        plt.title('ARIMA Model Evaluation')
        plt.xlabel('Time')
        plt.ylabel('Alt')
        plt.legend()
        plt.show()
        """
    
    results_df.to_csv("arima_benchmark_results.csv")

main()



