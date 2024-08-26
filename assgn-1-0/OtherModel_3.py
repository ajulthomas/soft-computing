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
        "distance_candle1",
        "distance_candle2",
        "distance_candle3",
        "distance_candle4",
        "Orientation_robot",
        "Collision",
    ]

    # target variables
    target = ["X_speed", "Y_speed"]


main()
