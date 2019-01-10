from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf


#DIRECTORY
#series = read_csv('C:/Users/aashi/Desktop/DataSet/Segment5/trainingdatatimeformat.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
"""#plot_pacf(series, lags=50)
autocorrelation_plot(series)
#series.plot()
pyplot.show()"""


# fit model
# test approaches
"""model = ARIMA(series, order=(10, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
print("mean is: ")
print(residuals.mean())
pyplot.show()
print(residuals.describe())"""



from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#DIRECTORY
series1 = read_csv('/inputdirectory', header=0, parse_dates=[0], index_col=0, squeeze=True) #training values
series2 = read_csv('/inputdirectory2', header=0, parse_dates=[0], index_col=0, squeeze=True) #testing values
#date_parser=parser)
X = series1.values
Y = series2.values
#size = int(len(X) * 0.66)
train, test = X[0:int(len(Y))], Y[0:int(len(Y))]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 0)) #order = (p,d,q) where p = lag observations, d = time differencing, q = window size
    model_fit = model.fit(disp=0)
    output = model_fit.forecast() #forecasting step.
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions) #results in "Segment5" folder
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
