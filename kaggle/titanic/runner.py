import os

import pandas as pd
import numpy as np
import sklearn
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# kaggle specific
from kaggle.utils.log_utils import get_default_logger

# titanic specific
import kaggle.titanic.feature as fe

logger = get_default_logger()


def load_data(csv_path, columns, dropna=False):
    df = pd.read_csv(csv_path)
    df = fe.extract_feature(df)
    df = df.loc[:, columns]
    logger.info(df.head())
    if dropna:
        df = df.dropna()
    return df

def split_train_eval(df, train_ratio):
    train_data, eval_data = np.split(df, [int(train_ratio*len(df))])
    return train_data, eval_data


def process_train_data(df, columns):
    df = df.loc[:, columns]
    return df


def eval(model, cost_func, y_true, y_pred):
    cost = cost_func(y_true, y_pred)
    return cost

def prepare_data(train_partition, y_column, columns):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Data
    # test_path = f"{dir_path}/data/test.csv"
    # test_data = load_data(test_path, columns, True)

    train_path = f"{dir_path}/data/train.csv"
    train_data = load_data(train_path, [y_column] + columns, True)
    train_partition, eval_partition = split_train_eval(train_data, train_partition)
    return train_partition, eval_partition

def run_linear_model(train_partition, eval_partition, y_column, columns):

    train_x = train_partition.loc[:, columns]
    train_y = train_partition.loc[:, [y_column]].values.reshape((len(train_partition),))

    model = LogisticRegression(C=1e5)
    model.fit(train_x, train_y)

    eval_x = eval_partition.loc[:, columns]
    eval_y = eval_partition.loc[:, [y_column]].values.reshape((len(eval_partition),))

    # Train and eval
    y_pred = model.predict(eval_x)
    y_true = eval_y
    logger.debug(f"Truth: {y_true}")
    logger.debug(f"Prediction: {y_pred}")

    cost = eval(model, mean_squared_error, y_true, y_pred)
    logger.info(f"MSE is {cost}")

def main():
    columns_ori = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    columns_new = ["SexDigit", "EmbarkedDigit"]

    columns_ticket = ["Ticket", "TicketNormalized", "TicketDigit"]
    columns_ticket = ["TicketDigit"]

    columns_cabin = ["CabinCode", "CabinNum", "CabinIndex", "Cabin"]
    columns_cabin = ["CabinCode", "CabinNum"]

    columns = columns_ori + columns_new + columns_ticket + columns_cabin
    y_column = "Survived"
    train_partition, eval_partition = prepare_data(0.8, y_column, columns)
    run_linear_model(train_partition, eval_partition, y_column, columns)

if __name__ == '__main__':
    main()
