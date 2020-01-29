''' Arima Modelling Functions '''

## Imports ##
# Standard Libraries #
import pandas as pd
import numpy as np
import itertools
import pickle
import io

# Visualisation Libraries #
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Libraries #
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

# Time Series Plots #
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Error Metrics #
from sklearn.metrics import mean_squared_error


# Seaborn Plotting Style #
sns.set(style='ticks', context='poster')

def dummy_creation(df, dayofweek_dummies=False, month_dummies=False):
    '''
    Aggregates the dataframe by Date and creates dummy variables.
    '''

    # Aggregating the entries by Date #
    df = df.groupby('Date').agg({'Sales':'sum',
                                 'DayOfWeek':'mean',
                                 'Month':'mean'})

    df['LogSales'] = np.log1p(df.Sales)
    
    # Day of Week dummy variables #
    if dayofweek_dummies:
        day_of_week_dummies = pd.get_dummies(df.DayOfWeek)
        day_of_week_dummies.columns = ['Monday', 'Tuesday', 'Wednesday',
                                       'Thursday', 'Friday', 'Saturday']

        df = df.merge(day_of_week_dummies, how='inner', on='Date')

    # Month dummy variables - default is False #
    if month_dummies:
        month_dummies = pd.get_dummies(df.Month)
        month_dummies.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        df = df.merge(month_dummies, how='inner', on='Date')

    df.drop('DayOfWeek', axis=1, inplace=True)
    df.drop('Month', axis=1, inplace=True)
    
    return df



def train_test_split(df, verbose=True):
    '''
    Function to split the dataframe into 2 sets of training and testing datasets for walk forward validation.
    '''

    trainset_1 = df[:'20141231']
    testset_1 = df['20150101':'20150331']
    trainset_2 = df[:'20150331']
    testset_2 = df['20150401':]

    if verbose:
        print('Initial dataset has been split into two sets of training and testing datasets each.')
        print('='*83)
        print('Trainset 1 is from 2013-01-02 to 2014-12-31 : N_obs = {}'.format(trainset_1.shape[0]))
        print('Testset 1 is from 2015-01-01 to 2015-03-31 : N_obs = {}'.format(testset_1.shape[0]))
        print('\n')
        print('Trainset 2 is from 2013-01-02 to 2015-03-31 : N_obs = {}'.format(trainset_2.shape[0]))
        print('Testset 2 is from 2015-04-01 to 2015-07-31 : N_obs = {}'.format(testset_2.shape[0]))

    return trainset_1, testset_1, trainset_2, testset_2



def differencing_series(df, diff_steps=1, split_data=True):
    '''
    Function that takes the first differences of the Sales and Logged Sales variables.
    By default, the funciton will split the new dataframe using the train_test_split function and returns 
    2 sets of training and testing datasets.
    '''
    # Taking first differences of both Sales and Logged Sales #
    df.Sales = df.Sales.diff(diff_steps)
    df.LogSales = df.LogSales.diff(diff_steps)

    # We lose observations from differencing #
    df.dropna(inplace=True)

    if split_data:
        trainset_1, testset_1, trainset_2, testset_2 = train_test_split(df)
        return trainset_1, testset_1, trainset_2, testset_2

    else:
        return df



def tsplot(y, lags=60, title='', figsize=(12, 12)):
    '''
    Plots the Time Series, ACF and PACF of the Series.

    :params:
      y (Series) : Series to be plotted.
      lags (int) : Number of lags to plot for the ACF and PACF graphs
      title (str) : Title of the time series graph for y.
      figsize : Figure size of the plot.

    :returns:
      ts_ax : Time series plot
      acf_ax : ACF plot
      pacf_ax : PACF plot

    '''

    fig = plt.figure(figsize=figsize)
    layout = (3, 1)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (2, 0))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    plot_acf(y, lags=lags, ax=acf_ax)
    plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()

    return ts_ax, acf_ax, pacf_ax


def adfuller_test(timeseries, title=''):
    '''
    Performs the Augmented Dickey Fuller test on the series to check for stationarity.
    '''

    print('The results for the Augmented Dickey-Fuller test for {}:\n'.format(title))

    adf_test = adfuller(timeseries, autolag='AIC')

    # Creating a new series for the test output #
    dfoutput = pd.Series(adf_test[0:4],index=['Test Statistic',
                                              'p-value',
                                              'Number of Lags Used',
                                              'Number of Observations'])

    for key,value in adf_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)


def optimise_SARIMAX(endog, exog=None, min_p=1, max_p=6, min_q=1, max_q=6, d=0,
                     min_P=0, max_P=5, min_Q=0, max_Q=5, D=0, S=6,
                     seasonal=False, verbose=True):
    '''
    Grid Search Function to find the optimal ARIMAX order using AIC.
    Returns a dataframe with the respective parameters and the associated AIC.

    :params:
    min_p (int) : Lower bound of the AR term. 
    max_p (int) : Upper bound of the AR term.
    min_q (int) : Lower bound of the MA term.
    max_q (int) : Upper bound of the MA term.
    d (int) : Order of differencing of the series. 

    :returns:
    result_table : Sorted dataframe with parameters (p, q) as indexes and AIC as values.  

    '''
    # Upper and lower bounds of parameter search #
    ps = range(min_p, max_p+1)
    qs = range(min_q, max_q+1)
    Ps = range(min_P, max_P+1)
    Qs = range(min_Q, max_Q+1)
    
    if seasonal:
        # Creating a list of all possible permuations of paramter combinations
        parameters = itertools.product(ps, qs, Ps, Qs)
        parameters_list = list(parameters)
        print('Total number of parameter permutations is : {}' \
              .format(len(parameters_list)))
    
    else:
        # Creating a list of all possible permuations of paramter combinations
        parameters = itertools.product(ps, qs)
        parameters_list = list(parameters)
        print('Total number of parameter permutations is : {}' \
              .format(len(parameters_list)))

    results = []
    best_aic = float('inf')

    for param in parameters_list:
        if seasonal:
            try:
                model = smt.statespace.SARIMAX(endog, exog,
                                               order=(param[0], d, param[1]),
                                               seasonal_order=(param[2], D, param[3], S)).fit(disp=-1)
            except:
                continue
        else:
            try: 
                model = smt.statespace.SARIMAX(endog, exog,
                                               order=(param[0], d, param[1])).fit(disp=-1)
            except:
                continue

        aic = model.aic

        # Saving the best model, AIC and parameters #
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    GS_results = pd.DataFrame(results)
    GS_results.columns = ['Parameters', 'AIC']

    # Sort in ascending order, with lower AIC being better #
    GS_results = GS_results.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    if verbose:
        print(GS_results.head(10))

    return GS_results


def model_selection(grid_search_results, endog, exog, d=0, D=0, S=0, model_index=0,
                    seasonal=False, verbose=True):
    '''
    The users selects a model from the Grid Search dataframe by setting the model_index.
    The function then returns the model and its associated parameters (p, q).

    :params:
      grid_search_results : Grid Search dataframe produced by optimize_SARIMAX function.
      endog (Series) : Target variable to fit in ARIMAX.
      exog (Dataframe) : Exogenous variables to fit in ARIMAX.
      d (int) : Order of differentiation of the target time series.
      model_index (int) : Index position of the selected ARIMAX parameters. Default is the parameters with the lowest AIC. 

    :returns:
      selected_model : statsmodels.SARIMAXResults object fitted with selected parameters.
      selected_order : Parameters of ARIMAX model chosen.

    '''
    
    # Set parameters that give the lowest AIC (Akaike Information Criteria) #
    
    if seasonal:
        p, q, P, Q = grid_search_results.Parameters[model_index]
        selected_order = (p, d, q)
        selected_seasonal_order = (P, D, Q, S)

        selected_model = smt.statespace.SARIMAX(endog, exog,
                                                order=selected_order,
                                                seasonal_order=selected_seasonal_order).fit(disp=-1)
        
        if verbose:
            print(selected_model.summary())

        return selected_model, selected_order, selected_seasonal_order
    
    p, q = grid_search_results.Parameters[model_index]
    selected_order = (p, d, q)

    selected_model = smt.statespace.SARIMAX(endog, exog, 
                                            order=selected_order).fit(disp=-1)
    
    if verbose:
        print(selected_model.summary())
    
    return selected_model, selected_order



def mean_absolute_percentage_error(y_true, y_pred, verbose=True):
    '''
    Function that calculates the Mean Absolute Percentage Error and displays it.
    '''
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    if verbose:
        print('The Mean Absolute Percentage Error is: {:.2f}%.'.format(MAPE))

    return MAPE



def root_mean_squared_error(y_true, y_pred, verbose=True):
    '''
    Function that calculates the Root Mean Squared Error and displays it.
    '''
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))

    if verbose:
        print('The Root Mean Squared Error is: {:.2f}.'.format(RMSE))

    return RMSE



def ARIMAX_forecast(trainset, testset, endog_name, best_arima_order, exog=False):

    forecasts = []
    history = trainset.copy()
    predictions_df = pd.DataFrame(testset['{}'.format(endog_name)])

    for index in range(testset.shape[0]):
        endog_var = history['{}'.format(endog_name)]
        if exog:
            exog_var_train = history.drop(['Sales', 'LogSales'], axis=1)
            exog_var_forecast = pd.DataFrame(testset.iloc[index, 2:]).transpose()
        
        else:
            exog_var_train = None
            exog_var_forecast=None

        model = smt.statespace.SARIMAX(endog_var,
                                       exog_var_train,
                                       order=best_arima_order).fit(disp=-1)

        # 1 Step ahead forecast, returns a series #
        forecast = model.forecast(steps=1,
                                  exog=exog_var_forecast)

        # Saving forecasts to a list #
        forecasts.append(forecast.iloc[0])

        # Appending entries from the testset to the training set (history) #
        history = history.append(testset.iloc[index, :])

    # Appending forecasts to dataframe #
    forecasts = np.array(forecasts)
    predictions_df['Forecasts'] = forecasts

    # Calculating MAPE and displaying results #
    MAPE = mean_absolute_percentage_error(predictions_df['{}'.format(endog_name)],
                                        predictions_df.Forecasts,
                                        verbose=True)

    RMSE = root_mean_squared_error(predictions_df['{}'.format(endog_name)], 
                                 predictions_df.Forecasts, 
                                 verbose=True) 

    return predictions_df, MAPE, RMSE



def SARIMAX_forecast(trainset, testset, endog_name,
                     best_arima_order, best_seasonal_order,
                     initialization='approximate_diffuse'):

    forecasts = []
    history = trainset.copy()
    predictions_df = pd.DataFrame(testset['{}'.format(endog_name)])

    for index in range(testset.shape[0]):
        endog_var = history['{}'.format(endog_name)]

        model = smt.statespace.SARIMAX(endog_var,
                                       order=best_arima_order,
                                       seasonal_order=best_seasonal_order,
                                       initialization='approximate_diffuse').fit(disp=-1)

        # 1 Step ahead forecast, returns a series #
        forecast = model.forecast(steps=1)

        # Saving forecasts to a list #
        forecasts.append(forecast.iloc[0])

        # Appending entries from the testset to the training set (history) #
        history = history.append(testset.iloc[index, :])

    # Appending forecasts to dataframe #
    forecasts = np.array(forecasts)
    predictions_df['Forecasts'] = forecasts

    # Calculating MAPE and displaying results #
    MAPE = mean_absolute_percentage_error(predictions_df['{}'.format(endog_name)],
                                        predictions_df.Forecasts,
                                        verbose=True)

    RMSE = root_mean_squared_error(predictions_df['{}'.format(endog_name)], 
                                 predictions_df.Forecasts, 
                                 verbose=True) 

    return predictions_df, MAPE, RMSE



def predictions_plot(y_true, y_pred, figsize=(8,4)):
    '''
    Plots the actual and predicted values on the same axis.
    '''
    MAPE = mean_absolute_percentage_error(y_true, y_pred, verbose=False)

    plt.figure(figsize=figsize)
    plt.title('Actual Sales vs Predicted Sales\nMAPE = {:.2f}%'.format(MAPE))

    actual_ax = y_true.plot(color='blue', grid=True, label='Actual Sales')

    forecast_ax = y_pred.plot(color='red',grid=True, label='Predicted Sales')



def process_data(df, dayofweek_dummies=False, month_dummies=False):
    
    df = dummy_creation(df, dayofweek_dummies=dayofweek_dummies,
                        month_dummies=month_dummies)
    
    trainset_1, testset_1, trainset_2, testset_2 = train_test_split(df)
    
    return trainset_1, testset_1, trainset_2, testset_2