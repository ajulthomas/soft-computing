# import the necessary packages
from data_loader import load_data
from feature_engineering import feature_engineering


def main():
    # define filepath
    file_path = "./data/RobotData.csv"

    # Load data
    df = load_data(file_path)

    # Perform feature engineering
    df_fe = feature_engineering(df)

    # print head of the dataframe
    print(df_fe.head())

    # selected features
    selected_features = [
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
    ]

    # target variables
    target = ["X_speed", "Y_speed"]


main()
