# function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# function that performs feature engineering on the dataset
def feature_engineering(robots_df):
    # calculating the distance between the robot and each candle
    robots_df["distance_candle1"] = euclidean_distance(
        robots_df["X_robot"],
        robots_df["Y_robot"],
        robots_df["X_candle1"],
        robots_df["Y_candle1"],
    )
    robots_df["distance_candle2"] = euclidean_distance(
        robots_df["X_robot"],
        robots_df["Y_robot"],
        robots_df["X_candle2"],
        robots_df["Y_candle2"],
    )
    robots_df["distance_candle3"] = euclidean_distance(
        robots_df["X_robot"],
        robots_df["Y_robot"],
        robots_df["X_candle3"],
        robots_df["Y_candle3"],
    )
    robots_df["distance_candle4"] = euclidean_distance(
        robots_df["X_robot"],
        robots_df["Y_robot"],
        robots_df["X_candle4"],
        robots_df["Y_candle4"],
    )

    # calculating the distance between the robot and each candle in each axis
    robots_df["distance_candle1_x"] = robots_df["X_robot"] - robots_df["X_candle1"]
    robots_df["distance_candle1_y"] = robots_df["Y_robot"] - robots_df["Y_candle1"]
    robots_df["distance_candle2_x"] = robots_df["X_robot"] - robots_df["X_candle2"]
    robots_df["distance_candle2_y"] = robots_df["Y_robot"] - robots_df["Y_candle2"]
    robots_df["distance_candle3_x"] = robots_df["X_robot"] - robots_df["X_candle3"]
    robots_df["distance_candle3_y"] = robots_df["Y_robot"] - robots_df["Y_candle3"]
    robots_df["distance_candle4_x"] = robots_df["X_robot"] - robots_df["X_candle4"]
    robots_df["distance_candle4_y"] = robots_df["Y_robot"] - robots_df["Y_candle4"]

    return robots_df
