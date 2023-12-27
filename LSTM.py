import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.src.layers import Bidirectional
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# 生成标签值：下一天收盘价（涉及删除最后一条数据，不要重复执行该函数）
def generate_label(data_path):
    df = pd.read_csv(data_path)
    next_close = list()
    for i in range(len(df['Close']) - 1):
        next_close.append(df['Close'][i + 1])
    next_close.append(0)
    df['next_close'] = next_close
    df.drop(df.index[-1], inplace=True)
    df.to_csv('temp.csv', index=None)


# 生成训练和测试数据
def generate_model_data(data_path, alpha, days):
    df = pd.read_csv(data_path)
    train_day = int((len(df['Close']) - days + 1))
    for property in ['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close']:
        df[property] = scaler.fit_transform(np.reshape(np.array(df[property]), (-1, 1)))
    X_data, Y_data = list(), list()
    # 生成时序数据
    for i in range(train_day):
        Y_data.append(df['Close'][i + days - 1])
        for j in range(days):
            for m in ['Open', 'Adj Close', 'High', 'Low', 'Volume']:
                X_data.append(df[m][i + j])
    X_data = np.reshape(np.array(X_data), (-1, 5 * days))  # 5表示特征数量*天数
    train_length = int(len(Y_data) * alpha)
    X_train = np.reshape(np.array(X_data[:train_length]), (len(X_data[:train_length]), days, 5))
    X_test = np.reshape(np.array(X_data[train_length:]), (len(X_data[train_length:]), days, 5))
    Y_train, Y_test = np.array(Y_data[:train_length]), np.array(Y_data[train_length:])
    return X_train, Y_train, X_test, Y_test


def calc_MAPE(real, predict):
    Score_MAPE = 0
    for i in range(len(predict[:, 0])):
        Score_MAPE += abs((predict[:, 0][i] - real[:, 0][i]) / real[:, 0][i])
    Score_MAPE = Score_MAPE * 100 / len(predict[:, 0])
    return Score_MAPE


def calc_AMAPE(real, predict):
    Score_AMAPE = 0
    Score_MAPE_DIV = sum(real[:, 0]) / len(real[:, 0])
    for i in range(len(predict[:, 0])):
        Score_AMAPE += abs((predict[:, 0][i] - real[:, 0][i]) / Score_MAPE_DIV)
    Score_AMAPE = Score_AMAPE * 100 / len(predict[:, 0])
    return Score_AMAPE


def evaluate(real, predict):
    RMSE = math.sqrt(mean_squared_error(real[:, 0], predict[:, 0]))
    MAE = mean_absolute_error(real[:, 0], predict[:, 0])
    MAPE = calc_MAPE(real, predict)
    AMAPE = calc_AMAPE(real, predict)
    return RMSE, MAE, MAPE, AMAPE


def lstm_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(units=20, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=200, batch_size=20, verbose=1)

    trainPredict = model.predict(X_train)
    trainPredict = scaler.inverse_transform(trainPredict)
    Y_train = scaler.inverse_transform(np.reshape(Y_train, (-1, 1)))

    testPredict = model.predict(X_test)
    testPredict = scaler.inverse_transform(testPredict)
    Y_test = scaler.inverse_transform(np.reshape(Y_test, (-1, 1)))

    return Y_train, trainPredict, Y_test, testPredict

def backtest(predictions, original_prices, threshold, initial_cash):
    cash = initial_cash
    shares = 0
    total_assets = initial_cash

    for i in range(1, len(predictions)):
        predicted_change = (predictions[i] - original_prices[i - 1])/ original_prices[i - 1]

        # Decision to buy
        if predicted_change > threshold and cash >= original_prices[i]:
            shares_bought = cash // original_prices[i]
            shares += shares_bought
            cash -= shares_bought * original_prices[i]

        # Decision to sell
        elif predicted_change < -threshold and shares > 0:
             cash += shares * original_prices[i]
             shares = 0

        # Update total assets value
        total_assets = cash + shares * original_prices[i]

    total_return = total_assets - initial_cash
    return total_assets, total_return

if __name__ == '__main__':
    data_path = 'AMZN_stock_data.csv'
    days = 30
    alpha = 0.8
    generate_label(data_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, Y_train, X_test, Y_test = generate_model_data(data_path, alpha, days)
    train_Y, trainPredict, test_Y, testPredict = lstm_model(X_train, Y_train, X_test, Y_test)
    plt.plot(list(trainPredict), color='red', label='prediction')
    plt.plot(list(train_Y), color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('train data')
    plt.show()
    plt.plot(list(testPredict), color='red', label='prediction')
    plt.plot(list(test_Y), color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.show()

    RMSE, MAE, MAPE, AMAPE = evaluate(test_Y, testPredict)
    print(RMSE, MAE, MAPE, AMAPE)

    initial_cash = 500000
    threshold = 0.001
    # Applying the backtest function to the LSTM model's predictions
    final_assets, total_return = backtest(testPredict[:, 0], test_Y[:, 0], threshold=threshold,
                                          initial_cash=initial_cash)
    return_rate = (total_return / initial_cash) * 100

    print("最终资产:", final_assets, "总回报:", total_return, "回报率:", return_rate)