# import the necessary packages
import pickle
from data_loader import load_data
from feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neural_network import MLPRegressor


def main(mode="develop"):
    # define filepath
    file_path = "./data/RobotData.csv"

    # Load data
    df = load_data(file_path)

    # data augmentation needs to be done here, if required

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
        "distance_candle1_x",
        "distance_candle1_y",
        "distance_candle2_x",
        "distance_candle2_y",
        "distance_candle3_x",
        "distance_candle3_y",
        "distance_candle4_x",
        "distance_candle4_y",
    ]

    # target variables
    target = ["X_speed", "Y_speed"]

    # get feature and target vectors
    X = df_fe[selected_features]
    y = df_fe[target]

    # scale the data
    if mode == "develop":
        develop(X, y)
    else:
        evaluate(X, y)


def develop(X, y):
    # split the data into training, validation and test sets
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_validate, y_train_validate, test_size=0.125, random_state=42
    )

    # scale the data
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # apply PCA to reduce the dimensionality of the data
    pca = PCA(n_components=0.95).fit(X_train_scaled)
    x_train_pca = pca.transform(X_train_scaled)
    x_validate_pca = pca.transform(X_validate_scaled)
    x_test_pca = pca.transform(X_test_scaled)

    # train the model
    hyper_params = {
        # "hidden_layer_sizes": (100, 100),
        # "activation": "relu",
        # "solver": "adam",
        # "max_iter": 1000,
        # "random_state": 33,
        # "batch_size": 32,
        # "learning_rate_init": "0.001",
        "verbose": False,
    }
    model = MLPRegressor(**hyper_params).fit(x_train_pca, y_train)

    # evaluate the model
    train_score = model.score(x_train_pca, y_train)
    validate_score = model.score(x_validate_pca, y_validate)
    test_score = model.score(x_test_pca, y_test)

    # model props
    model_props = {
        "id": id,
        "hidden_layers": model.n_layers_ - 2,
        "architecture": model.hidden_layer_sizes,
        "activation": model.activation,
        "batch_size": model.batch_size,
        "solver": model.solver,
        "learning_rate_init": model.learning_rate_init,
        "learning_rate": model.learning_rate,
        "iterations": model.n_iter_,
        "train_score": train_score,
        "val_score": validate_score,
        "test_score": test_score,
    }

    # print the model properties
    print(model_props)

    # save the scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # save the pca
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    # save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def evaluate(X, y):
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    pca = pickle.load(open("pca.pkl", "rb"))

    # scale the data
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # apply PCA to reduce the dimensionality of the data
    x_pca = pca.transform(X_scaled)

    # evaluate the model
    test_score = model.score(x_pca, y)

    # model props
    model_props = {
        "id": id,
        "hidden_layers": model.n_layers_ - 2,
        "architecture": model.hidden_layer_sizes,
        "activation": model.activation,
        "batch_size": model.batch_size,
        "solver": model.solver,
        "learning_rate_init": model.learning_rate_init,
        "learning_rate": model.learning_rate,
        "iterations": model.n_iter_,
        "test_score": test_score,
    }

    # print the model properties
    print(model_props)


main()
