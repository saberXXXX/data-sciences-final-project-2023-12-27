import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# 生成标签值：下一天收盘价
def generate_label(data_path):
    df = pd.read_csv(data_path)
    df['next_close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df.to_csv('temp.csv', index=None)


# 生成线性回归模型的特征和标签
def generate_model_data(data_path):
    df = pd.read_csv(data_path)
    X = df[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
    y = df['next_close']
    return X, y


if __name__ == '__main__':
    data_path = 'AAPL_stock_data.csv'
    generate_label(data_path)

    X, y = generate_model_data('temp.csv')

    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 初始化并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 可视化结果
    plt.plot(y_test.values, label='real')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.legend()
    plt.title('Linear Regression - Test Set')
    plt.show()

    # 评估模型性能
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)