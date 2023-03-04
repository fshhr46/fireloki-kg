import pandas as pd
import numpy as np
import sklearn
import matplotlib
from sklearn.linear_model import LogisticRegression


def sex_to_digit(sex: str):
    if sex == 'male':
        return 1
    else:
        return 0


def load_data(csv_path, columns, dropna=False):
    df = pd.read_csv(csv_path)
    df['SexDigit'] = df['Sex'].map(sex_to_digit)
    df = df.loc[:, columns]
    if dropna:
        df = df.dropna()
    return df


def split_train_eval(df, train_ratio):
    train_data, eval_data = np.split(df, [int(train_ratio*len(df))])
    return train_data, eval_data


def process_train_data(df, columns):
    df = df.loc[:, columns]
    return df


def train(data, label):
    logreg = LogisticRegression(C=1e5)
    logreg.fit(data, label)
    return logreg


def main():
    test_path = "data/test.csv"
    test_data = load_data(test_path, ["SexDigit", "Age"], True)

    train_path = "data/train.csv"
    train_data = load_data(train_path, ["Survived", "SexDigit", "Age"], True)
    train_partition, eval_partition = split_train_eval(train_data, 0.8)

    train_x = train_partition.loc[:, ["SexDigit", "Age"]]
    train_y = train_partition.loc[:, ["Survived"]]
    model = train(train_x, train_y)

    eval_x = eval_partition.loc[:, ["SexDigit", "Age"]]
    eval_y = eval_partition.loc[:, ["Survived"]]
    pred_y = model.predict(eval_x)
    print(f"Truth: {eval_y.to_numpy().flatten()}")
    print(f"Prediction: {pred_y}")


if __name__ == '__main__':
    main()
