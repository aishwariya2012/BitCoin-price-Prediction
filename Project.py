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
price = series.iloc[:, [6]].fillna(method ='ffill')
open = series.iloc[:, [0]].fillna(method ='ffill')
high = series.iloc[:, [1]].fillna(method = 'ffill')
low =  series.iloc[:, [2]].fillna(method = 'ffill')
close = series.iloc[:, [3]].fillna(method = 'ffill')

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
openValues = open.values
highValues = high.values
lowValues = low.values
closevalues = close.values
size = int(len(X) * 0.66)
series = read_csv('CMPE-256-Large-Scale-Analytics-/data/bitstamp.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# X = series.iloc[:,[6]].fillna(method = 'ffill').head(1000).values
price = series.iloc[:, [6]].fillna(method='ffill').head(1000).values
open = series.iloc[:, [0]].fillna(method='ffill').head(1000).values
high = series.iloc[:, [1]].fillna(method='ffill').head(1000).values
low = series.iloc[:, [2]].fillna(method='ffill').head(1000).values
close = series.iloc[:, [3]].fillna(method='ffill').head(1000).values
size = int(len(price) * 0.66)
trainPrice, testPrice = price[0:size], price[size:len(price)]
trainOpen, testOpen = open[0:size], open[size: len(price)]
trainHigh, testHigh = high[0:size], high[size: len(price)]
trainLow, testLow = low[0:size], low[size: len(price)]
trainClose, testClose = close[0:size], close[size: len(price)]
historyPrice = [x for x in trainPrice]
historyOpen = [x for x in trainOpen]
historyHigh = [x for x in trainHigh]
historyLow = [x for x in trainLow]
historyClose = [x for x in trainClose]
predictions = list()
# print (testOpen)
testLen = len(testPrice)
for t in range(testLen):
    priceModel = ARIMA(historyPrice, order=(5, 1, 0)).fit(disp=0)
    openModel = ARIMA(historyOpen, order=(5, 1, 0)).fit(disp=0)
    highModel = ARIMA(historyHigh, order=(5, 1, 0)).fit(disp=0)
    lowModel = ARIMA(historyLow, order=(5, 1, 0)).fit(disp=0)
    closeModel = ARIMA(historyClose, order=(5, 1, 0)).fit(disp=0)
    # model_fit = model.fit(disp=0)
    outputPrice = priceModel.forecast()
    outputOpen = openModel.forecast()
    outputHigh = highModel.forecast()
    outputLow = lowModel.forecast()
    outputClose = closeModel.forecast()
    predict = list()
    predict.append(outputPrice[0])
    predict.append(outputOpen[0])
    predict.append(outputHigh[0])
    predict.append(outputLow[0])
    predict.append(outputClose[0])
    # yhat = output[0]
    predictions.append(predict)
    # predict = list()
    outputTest = list()
    outputTest.append(testPrice[t])
    outputTest.append(testOpen[t])
    outputTest.append(testHigh[t])
    outputTest.append(testLow[t])
    outputTest.append(testClose[t])
    # history.append(obs)
    print('predicted=%f %f %f, expected=%f %f %f' % (
        predict[0], predict[1], predict[2], outputTest[0], outputTest[1], outputTest[2], outputTest[3], outputTest[4]))
#     print('predicted')
#     print(predict)
# error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
# pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
