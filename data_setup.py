import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    # 读取数据
    data = pd.read_excel('data/Physical/SWaT_Dataset_Attack_v0.xlsx')
    # data = pd.read_excel('data/Physical/test.xlsx')

    # 删除日期列
    data = data.drop(columns=['Timestamp'])

    # 准备一个空的数据框来保存新的数据
    new_data = pd.DataFrame()

    # 设置序列长度
    sequence_length = 10

    # 遍历每一列
    for column in data.columns[:-1]:  # Excluding the last column (Normal/Attack)
        # 对于每一列，创建一个新的列，这个列的值是原列的值向上偏移i行
        for i in range(sequence_length):
            new_data[column+'_'+str(i)] = data[column].shift(i)

    # 删除包含空值的行
    new_data = new_data.dropna()

    # 把数据框转化为numpy数组
    X = new_data.values

    # 改变数组的形状
    X = X.reshape((X.shape[0], sequence_length, -1))

    # 分离出标签
    Y = data["Normal/Attack"]

    # 对标签进行编码
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    # Delete the same rows in Y as in new_data
    Y = Y[new_data.index]

    # 对数据进行划分
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
