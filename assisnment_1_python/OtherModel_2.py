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
        "distance_candle1_x",
        "distance_candle1_y",
        "distance_candle2_x",
        "distance_candle2_y",
        "distance_candle3_x",
        "distance_candle3_y",
        "distance_candle4_x",
        "distance_candle4_y",
        "Orientation_robot",
        "Collision",
    ]

    # target variables
    target = ["X_speed", "Y_speed"]


main()
