# Test and evaluate ARIMA model on 2000 different trials of F16 flight simulator
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_GCAS import main_run_gcas
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

def main():
    trial_nums = []
    trial_rmse = []
    for i in range(2000): # 2000 trials
        df = main_run_gcas() # Run flight simulator which uses randomized initial parameters
        adf_test = adfuller(df['alt']) # Test if data is stationary
        p_val_adf = adf_test[1]
        while(p_val_adf >= 0.05): # If not stationary, differentiate until it is
            df = df.diff().dropna()
            adf_test = adfuller(df['alt'])
            p_val_adf = adf_test[1]

        train_size = int(0.7*len(df)) # Split train and test
        train = df.iloc[0:train_size]
        test = df.iloc[train_size:]
        model = auto_arima(train['alt'], start_p=1, start_q=1, # Find best parameters for ARIMA model
                      test='adf',
                      max_p=10, max_q=10,
                      m=1,             
                      d=1,          
                      seasonal=False,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
        prediction, confint = model.predict(len(test), return_conf_int=True) # Make predictions
        cf= pd.DataFrame(confint)
        mse = mean_squared_error(test['alt'], prediction)
        rmse = mse**0.5
        trial_nums.append(i)
        trial_rmse.append(rmse) # Calculate RMSE for predictions for this trial


        plt.figure(figsize=(14,7)) # Plot prediction against test data
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
   
    rmse_trials = pd.DataFrame(columns = ['trialnumber', 'rmse'])
    rmse_trials['trialnumber'] = trial_nums
    rmse_trials['rmse'] = trial_rmse

    # Find best and worst RMSE values across the 2000 trials
    print("Best 5 RMSE Values:")
    print(df.nsmallest(5, 'rmse'))

    print("Worst 5 RMSE Values:")
    print(df.nlargest(5, 'rmse'))

main()



