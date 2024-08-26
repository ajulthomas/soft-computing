import pandas as pd


def load_data(filepath):
    # setting column names of the dataframe
    headers = [
        "X_robot",
        "Y_robot",
        "Orientation_robot",
        "Collision",
        "X_candle1",
        "Y_candle1",
        "X_candle2",
        "Y_candle2",
        "X_candle3",
        "Y_candle3",
        "X_candle4",
        "Y_candle4",
        "X_speed",
        "Y_speed",
    ]

    # load data from file
    df = pd.read_csv(filepath, names=headers)

    return df
