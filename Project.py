from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

def parser(x):
    dateout = []
    for time in x:
        dateout.append(datetime.fromtimestamp(int(time)))
    return dateout


series = read_csv('data/bitstamp.csv', header=0,
                  parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
price = series.iloc[:, [6]].fillna(method='ffill')
open = series.iloc[]

model = ARIMA(price, order=(5, 1, 0), missing='nan')
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# price = read_csv('data/bitstamp.csv', header=0,parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# X = price.iloc[:, [6]].fillna(method='ffill').head(100000).values
X = price.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# print (history)
for t in range(len(test)):
	model = ARIMA(history, order=(5, 1, 0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
