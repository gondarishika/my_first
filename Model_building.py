# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:23:42 2024

@author: rishika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
user = 'root'
pw = 'lapapappi123'
db = 'medical_inventory'

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data = pd.read_csv(r"C:\Users\rishika\Desktop\PROJECT\Medical Inventory Optimaization Dataset.csv")
data.to_sql('medical_data', con = engine,if_exists='replace',chunksize=1000,index=False)
sql = "select * from medical_data"
data = pd.read_sql_query(sql, engine)
data.info()
data.describe()
######################### EDA business moments#####################
data_mean = data.mean()
data_mean
data_median = data.median()
data_median
data_var = data.var()
data_var
data_std = data.std()
data_std
numerical_columns = data.select_dtypes(include='number')
data_range = numerical_columns.max()-numerical_columns.min()
data_range
data_skew = data.skew()
data_skew
data_kurt = data.kurtosis()
data_kurt
#######################TYPE-CASTING#################################
data['Patient_ID'] = data['Patient_ID'].astype(str)
data['Dateofbill'] = data['Dateofbill'].astype('datetime64')
data["Final_Sales"] = data["Final_Sales"].astype('float32')
data["Final_Cost"] = data["Final_Cost"].astype('float32')
data.info()

########################FINDING-DUPLICATES##########################

duplicate = data.duplicated()  
sum(duplicate)
# Remove duplicates
data = data.drop_duplicates() 
duplicate = data.duplicated()
sum(duplicate) 
########################MISSING VALUES##############################
data.isnull().sum()
data_col=['Formulation','DrugName','SubCat','SubCat1']
for col in data_col:
    data[col].fillna(data[col].mode()[0],inplace=True)
    data.isnull().sum()
    
########################OUTLIERS-TREATMENT###########################
data.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))
numerical_columns = ['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
for i in numerical_columns:
    IQR=data[i].quantile(0.75)-data[i].quantile(0.25)
    lower_limit=data[i].quantile(0.25)-(1.5*IQR)
    upper_limit=data[i].quantile(0.75)-(1.5*IQR)
    data[i]=pd.DataFrame(np.where(data[i]>upper_limit,upper_limit,np.where(data[i]<lower_limit,lower_limit,data[i])))
data.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))

###########################AFTER EDA#################################
######################### EDA business moments#####################
data_mean = data.mean()
data_mean
data_median = data.median()
data_median
data_var = data.var()
data_var
data_std = data.std()
data_std
numerical_columns = data.select_dtypes(include='number')
data_range = numerical_columns.max()-numerical_columns.min()
data_range
data_skew = data.skew()
data_skew
data_kurt = data.kurtosis()
data_kurt
import seaborn as sns
sns.barplot(data = data, x = 'Dateofbill', y = 'Quantity')
plt.title('Quantity of drugs sold by Month')
plt.show()

import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class 
import dtale

report_sweetviz = sv.analyze(data)
report_sweetviz.show_html('sweetviz_report2.html')

medical_data = data.copy()
medical_data.to_excel("C:/Users/rishika/Desktop/PROJECT/Cleandatafromnewfile.xlsx")

av = AutoViz_Class()
report_autoviz = av.AutoViz('C:/Users/rishika/Desktop/PROJECT/Medical Inventory Optimaization Dataset.csv')

d = dtale.show(data)
d.open_browser()

df_grouped = data[['Dateofbill','Quantity']]
# Group by Quantity and week
df_grouped = df_grouped.groupby('Dateofbill').sum()

# Result
df_grouped.head(10)
df_grouped = df_grouped.reset_index()
df_grouped

data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])
data['billof month'] = data['Dateofbill'].dt.month_name()
data.loc[:,'billof month'] = data['billof month'].str.slice(stop=3)
data_month = data.groupby('billof month')['Quantity'].sum().reset_index()
data_month
data['weekofbill'] = data['Dateofbill'].dt.isocalendar().week
data.reset_index(drop = True, inplace = True)
data_week = data.groupby('weekofbill')['Quantity'].sum().reset_index()
data_week
data2 = pd.get_dummies(data_week['weekofbill'], prefix='weekofbill')
data2

med_inv = pd.concat([data_week,data2],axis=1) 
med_inv
med_inv["t"] = np.arange(1,53) # linear trend
med_inv["t_square"] = med_inv["t"] * med_inv["t"]
med_inv["log_Quantity"] = np.log(med_inv["Quantity"])
med_inv
med_inv.to_excel("C:/Users/rishika/Desktop/PROJECT/NEW WEEK DUMMIES.xlsx")
import statsmodels.formula.api as smf
Train = med_inv
Test = med_inv
model_full = smf.ols('Quantity ~ t+t_square+weekofbill_1 + weekofbill_2 + weekofbill_3 + weekofbill_4 + weekofbill_5 + weekofbill_6 + weekofbill_7 + weekofbill_8 + weekofbill_9 + weekofbill_10 + weekofbill_11 + weekofbill_12 + weekofbill_13 + weekofbill_14+ weekofbill_15 + weekofbill_16 + weekofbill_17 + weekofbill_18 + weekofbill_19 + weekofbill_20 + weekofbill_21 + weekofbill_22 + weekofbill_23 + weekofbill_24 + weekofbill_25+ weekofbill_26 + weekofbill_27 + weekofbill_28 + weekofbill_29 + weekofbill_30+ weekofbill_31 + weekofbill_32 + weekofbill_33 + weekofbill_34 + weekofbill_35 + weekofbill_36 + weekofbill_37 + weekofbill_38 + weekofbill_39+ weekofbill_40 + weekofbill_41 + weekofbill_42 + weekofbill_43 + weekofbill_44 + weekofbill_45 + weekofbill_46+ weekofbill_47 + weekofbill_48+ weekofbill_49 + weekofbill_50 + weekofbill_51 + weekofbill_52',data=Train).fit()
predict_data = med_inv
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

def MAPE(pred,actual):
    temp = np.abs((pred-actual)/actual)*100
    return np.mean(temp)
#######################ARIMA MODEL############################

import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
########################AUTO-REGRESSION MODEL###################
full_res = med_inv.Quantity - model_full.predict(med_inv)
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA

tsa_plots.plot_acf(full_res, lags = 11)
tsa_plots.plot_pacf(full_res, lags = 5 )
######################ARIMA#####################################
train = data_week
test= data_week
train
tsa_plots.plot_acf(full_res, lags = 11)
tsa_plots.plot_pacf(full_res, lags = 5 )
model1 = ARIMA(train.Quantity, order = (5,1,2))
res1 = model1.fit()
res1.summary()
start_index = len(train)
start_index
end_index = start_index + 51
forecast_test = res1.predict(start = start_index, end = end_index)
forecast_test = pd.DataFrame(forecast_test)
forecast_test
from math import sqrt
from sklearn.metrics import mean_squared_error

rmse_test = sqrt(mean_squared_error(test.Quantity, forecast_test))

print('test RMSE: %.3f' % rmse_test)
pred_arima = res1.predict(start = test.index[0], end = test.index[-1])
ari = MAPE(pred_arima, test.Quantity)
ari

##########################SIMPLE EXPONENTIAL SMOOTHING #######################
# Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
ses_model = SimpleExpSmoothing(train["Quantity"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = Test.index[-1])
MAPE_SES_ExponentialSmoothing = MAPE(pred_ses, Test.Quantity) 
MAPE_SES_ExponentialSmoothing


################################HOLTS - WINTERS METHOD ########################
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing
# Holt method 
hw_model = Holt(train["Quantity"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = Test.index[-1])
hw_MAPE= MAPE(pred_hw, Test.Quantity) 
hw_MAPE

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Quantity"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = Test.index[-1])
hwe_MAPE = MAPE(pred_hwe_add_add, test.Quantity) 
hwe_MAPE

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Quantity"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = Test.index[-1])
hwe_w_MAPE = MAPE(pred_hwe_mul_add, test.Quantity)
hwe_w_MAPE

############################## MOVING - AVERAGE ###############################
mv_pred = med_inv["Quantity"].rolling(12).mean()
mv_pred.tail(4)
mv_MAPE = MAPE(mv_pred.tail(4), Test.Quantity)
mv_MAPE

############################ SARIMA ###########################################
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Assuming 'train_data' and 'test_data' are your time series DataFrames or NumPy arrays
# Replace 'column_name' with the actual column name in your DataFrames

# Extract the target column
train_series = train['Quantity'] if isinstance(train, pd.DataFrame) else train
test_series = test['Quantity'] if isinstance(test, pd.DataFrame) else test

# Set the order and seasonal_order parameters based on your data and requirements
order = (1, 1, 1)  # Example order (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Example seasonal order (P, D, Q, S)

# Fit the SARIMA model
sarima_model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit()

# Make predictions on test data
sarima_predictions = sarima_result.get_forecast(steps=len(test_series)).predicted_mean.values

# Calculate MAPE
sari_MAPE = mean_absolute_error(test_series, sarima_predictions) / np.mean(test_series) * 100
sari_MAPE
'''
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
train = med_inv
test=med_inv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(train['Quantity'])
plot_pacf(train['Quantity'])
sarima_model = SARIMAX(train['Quantity'], order=(1, 0, 1), seasonal_order=(0, 1, 1, 52))
sarima_result = sarima_model.fit()
pred_sarima = sarima_result.predict(start = test.index[0], end = test.index[-1])
sari_MAPE = MAPE(pred_sarima, test.Quantity)
sari_MAPE'''

############################ SARIMAX #########################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,RobustScaler
from sklearn.compose import ColumnTransformer
sarimax_data = data.drop(['Patient_ID','billof month'],axis =1)

sarimax_data.drop(['Typeofsales','DrugName','RtnMRP','Final_Sales','Dateofbill'], axis =1, inplace=True)
# sarimax_data.drop(['SubCat'], axis =1, inplace=True) When we drop this feature i got MAPE = 5.89% 
# and it not drops this i got the 2 approximately
sarimax_data.drop(['Specialisation'], axis =1, inplace=True)
sarimax_data.drop(['Quantity'], axis =1, inplace=True)
sarimax_data.info()

numerical = sarimax_data.select_dtypes(['int64','float64']).columns
categorical = sarimax_data.select_dtypes(['object']).columns
num = Pipeline(steps= [('scaling',RobustScaler())])
cat = Pipeline([('encoding',OneHotEncoder())])
preprocess = ColumnTransformer([('scaling',num,numerical),
                                ('encoding',cat,categorical)],remainder=  'passthrough')
preprocess_fit =  preprocess.fit(sarimax_data)


sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray() ,columns=preprocess_fit.get_feature_names_out())

sarimax_data = pd.concat([sarimax_data_preprocess,data[['Dateofbill','Quantity']]],axis =1)

sarimax_data['weekofbill'] = sarimax_data['Dateofbill'].dt.isocalendar().week
data['week_of_first_day'] = sarimax_data.groupby('weekofbill')['Dateofbill'].transform('min').dt.isocalendar().week
sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()



train = sarimax_data
test = sarimax_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarimax_model = SARIMAX(train['Quantity'],exog=train.drop(['Quantity','weekofbill'],axis=1), order=(4, 1, 10),seasonal_order=(1,0,1,52))
sarimax_result = sarimax_model.fit()
pred_sarimax = sarimax_result.predict(start = test.index[0], end = test.index[-1])
sarimax_MAPE = MAPE(pred_sarimax, test.Quantity)
sarimax_MAPE  #
'''from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
# Extract the target column
train_series = train['Quantity']
test_series = test['Quantity']

# Set the order and seasonal_order parameters based on your data and requirements
order = (1, 1, 1)  # Example order (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Example seasonal order (P, D, Q, S)

# Fit the SARIMAX model without exogenous variables
sarimax_model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
sarimax_result = sarimax_model.fit()

# Make predictions on test data
sarimax_predictions = sarimax_result.get_forecast(steps=len(test_series)).predicted_mean

# Calculate MAPE
sarimax_MAPE = mean_absolute_error(test_series, sarimax_predictions) / np.mean(test_series) * 100
sarimax_MAPE'''
'''
sarimax_model = SARIMAX(train['Quantity'],exog=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result = sarima_model.fit()
pred_sarimax = sarimax_result.predict(start = test.index[0], end = test.index[-1])
sarimax_MAPE= MAPE(pred_sarimax, test.Quantity)
sarimax_MAPE'''

############################# GRU Model #############################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values
# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))
# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)
# Set sequence length and create sequences for training
sequence_length = 10  # Adjust as needed
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]
#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
# Create a GRU model
model = Sequential()
model.add(GRU(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)
# Create sequences for testing
X_test = create_sequences(test_series_scaled, sequence_length)
# Make predictions
predictions_scaled = model.predict(X_test)
# Invert the scaling to get actual predictions
predictions = scaler.inverse_transform(predictions_scaled)
# Calculate MAPE
def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100
GRU_MAPE = calculate_mape(test_series[sequence_length:], predictions.flatten())
GRU_MAPE
###########################FTS MODEL ########################################
#pip install scikit-fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error
# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Define fuzzy sets and antecedents
antecedent = ctrl.Antecedent(np.arange(min(train_series), max(train_series), 1), 'input')
consequent = ctrl.Consequent(np.arange(min(train_series), max(train_series), 1), 'output')

# Create fuzzy membership functions
antecedent['low'] = fuzz.trimf(antecedent.universe, [min(train_series), min(train_series), np.median(train_series)])
antecedent['medium'] = fuzz.trimf(antecedent.universe, [min(train_series), np.median(train_series), max(train_series)])
antecedent['high'] = fuzz.trimf(antecedent.universe, [np.median(train_series), max(train_series), max(train_series)])

consequent['low'] = fuzz.trimf(consequent.universe, [min(train_series), min(train_series), np.median(train_series)])
consequent['medium'] = fuzz.trimf(consequent.universe, [min(train_series), np.median(train_series), max(train_series)])
consequent['high'] = fuzz.trimf(consequent.universe, [np.median(train_series), max(train_series), max(train_series)])

# Create fuzzy rules
rule1 = ctrl.Rule(antecedent['low'], consequent['low'])
rule2 = ctrl.Rule(antecedent['medium'], consequent['medium'])
rule3 = ctrl.Rule(antecedent['high'], consequent['high'])
# Create Fuzzy Control System
fts_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fts_simulation = ctrl.ControlSystemSimulation(fts_ctrl)
# Fit the model with training data
fts_simulation.input['input'] = train_series
fts_simulation.compute()
# Make predictions on test data
fts_predictions = fts_simulation.output['output']
# Calculate MAPE
FTS_MAPE = mean_absolute_error(test_series, fts_predictions) / np.mean(test_series) * 100
FTS_MAPE
#############################VAR MODEL ######################################
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error

VAR_data = med_inv[['Quantity','t_square', 'log_Quantity']]
model = VAR(VAR_data)
model_fitted = model.fit()
lag_order = model_fitted.k_ar

forecast = model_fitted.forecast(VAR_data.values[-lag_order:], steps=10)
VAR_mape = MAPE(VAR_data[-10:], forecast)
VAR_mape

############################LONG SHORT TERM MEMORY MODEL ######################
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))

# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Set sequence length
sequence_length = 10  # Adjust as needed

# Create sequences for training
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]

# Create sequences for testing
X_test = create_sequences(test_series_scaled, sequence_length)

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on test data
predictions_scaled = model.predict(X_test)

# Invert the scaling to get actual predictions
predictions = scaler.inverse_transform(predictions_scaled)

# Evaluate the model (you may use other metrics like MSE, MAE, etc.)
LSTM_MAPE = mean_absolute_error(test_series[sequence_length:], predictions.flatten()) / np.mean(test_series) * 100
LSTM_MAPE
##################################### RNN MODEL #############################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))

# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Set sequence length
sequence_length = 10  # Adjust as needed

# Create sequences for training
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]

# Create sequences for testing
X_test = create_sequences(test_series_scaled, sequence_length)

# Reshape data for RNN input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on test data
predictions_scaled = model.predict(X_test)

# Invert the scaling to get actual predictions
predictions = scaler.inverse_transform(predictions_scaled)

# Evaluate the model (you may use other metrics like MSE, MAE, etc.)
RNN_MAPE = mean_absolute_error(test_series[sequence_length:], predictions.flatten()) / np.mean(test_series) * 100
RNN_MAPE

comparing_mape=pd.Series({'ARIMA':ari,'SIMPLE EXPONENTIAL SMOOTHING':MAPE_SES_ExponentialSmoothing,'HOLTS EXPONENTIAL SMOOTHING' : hw_MAPE,'HOLTS SMOOTHING WITH ADDITIVE SEASONALITY':hwe_MAPE,'HS-MULTIPLICATIVE SEASONALITY':hwe_w_MAPE,'MOVING-AVG':mv_MAPE,
                          'SARIMA':sari_MAPE,'SARIMAX':sarimax_MAPE,'GRU-MODEL':GRU_MAPE,'FTS-MODEL':FTS_MAPE,'VAR-MODEL':VAR_mape,'LSTM-MODEL': LSTM_MAPE,'RNN-MODEL':RNN_MAPE})
mape = pd.DataFrame(comparing_mape, columns=['mape'])
mape
'''
################################### PROPHET MODEL ############################
#pip install prophet
#pip install fbprophet

from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error

# Assuming 'train_data' and 'test_data' are your time series DataFrames
# Replace 'ds' and 'y' with the actual column names in your DataFrames

# Extract the target column and rename columns to 'ds' and 'y' as required by Prophet
train_df = train[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'})
test_df = test[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'})

# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(train_df)

# Create a DataFrame for future dates (test period)
future = prophet_model.make_future_dataframe(periods=len(test_df))

# Generate forecasts for the future dates
forecast = prophet_model.predict(future)

# Extract actual values and predictions for the test period
actual_values = test_df['y'].values
predicted_values = forecast.iloc[-len(test_df):]['yhat'].values

# Calculate MAPE
mape = mean_absolute_error(actual_values, predicted_values) / np.mean(actual_values) * 100
print(f'MAPE: {mape:.2f}%')'''

##############################AUTO MODEL-BUILDING##############################
#pip install pmdarima
import pmdarima as pm
import numpy as np

# Generate sample time series data (2D array with 3 time series)
np.random.seed(42)
data = np.random.rand(100, 3)

# Apply auto_arima to each time series
for i in range(data.shape[1]):
    time_series = data[:, i]
    model = pm.auto_arima(time_series, seasonal=True, stepwise=True, suppress_warnings=True)
    print(f"Summary for Time Series {i+1}:\n{model.summary()}\n")
    
################################ TPOT #########################################
'''from tpot import TPOTRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate some placeholder data
X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the TPOT Regressor model
tpot = TPOTRegressor(generations=5, population_size=20, random_state=42, verbosity=2)
tpot.fit(X_train, y_train)

# Evaluate the best pipeline on the test set
test_score = tpot.score(X_test, y_test)
print(f"Model Score on Test Set: {test_score}")

# Export the pipeline code if needed
tpot.export('best_pipeline.py')'''
############################SAVING MODEL#######################################
import joblib
best_model = sarimax_result
joblib.dump(best_model, 'best_model_sarimax.pkl')
joblib.dump(preprocess_fit,'preprocess_fit.pkl')