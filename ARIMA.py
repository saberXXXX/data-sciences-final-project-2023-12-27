import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
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
    df['Adj Close'] = scaler.fit_transform(np.reshape(np.array(df['Adj Close']), (-1, 1)))

    Y_data = df['Adj Close'].values

    train_length = int(len(Y_data) * alpha)
    Y_train, Y_test = Y_data[:train_length], Y_data[train_length:]

    return Y_train, Y_test


def arima_forecast(train_data, test_data, order):
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test_data))
    return predictions


def evaluate(real, predict):
    RMSE = math.sqrt(mean_squared_error(real, predict))
    MAE = mean_absolute_error(real, predict)
    MAPE = np.mean(np.abs((real - predict) / real)) * 100
    AMAPE = np.mean(np.abs((predict - real) / np.mean(real))) * 100
    return RMSE, MAE, MAPE, AMAPE


if __name__ == '__main__':
    data_path = 'AAPL_stock_data.csv'
    days = 15
    alpha = 0.8
    generate_label(data_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    Y_train, Y_test = generate_model_data('AAPL_stock_data.csv', alpha, days)

    # 使用 pmdarima 进行自动选择 ARIMA 模型的 order 参数
    auto_arima_model = auto_arima(Y_train, seasonal=False, suppress_warnings=True, trace=True)
    order = auto_arima_model.order

    # 使用 ARIMA 进行预测
    test_predictions = arima_forecast(Y_train, Y_test, order)

    # 反归一化
    Y_test = scaler.inverse_transform(np.reshape(Y_test, (-1, 1)))
    test_predictions = scaler.inverse_transform(np.reshape(test_predictions, (-1, 1)))

    # 评估和可视化
    RMSE, MAE, MAPE, AMAPE = evaluate(Y_test, test_predictions)
    print(RMSE, MAE, MAPE, AMAPE)

    plt.plot(test_predictions, color='red', label='prediction')
    plt.plot(Y_test, color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.show()