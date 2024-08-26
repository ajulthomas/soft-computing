
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute  import KNNImputer


#This function is used if one or more of your column has categorical data
def convert_categorical_to_numeric(df, label_encoders, mode):
    """
    Parameters: df the input DataFrame with some categorical columns, list of label encoders, mode ('train' or 'test').
    Returns: df_converted with all columns converted to numeric types.
    """
    # Create a copy of df to avoid modifying the original one
    df_converted = df.copy()
    # for each column in df
    for column in range(len(df_converted.columns)):
        # Initialise LabelEncoder that performs integer encoding as described in the lecture
        if mode == 'develop':
          label_encoders.append(LabelEncoder())
        # Check if the column is of type object (string/categorical)
        if df_converted.iloc[:,column].dtype == 'object':
          # if so, conert using integer encoding
          if mode == 'develop':
            df_converted.iloc[:,column] = label_encoders[column].fit_transform(df_converted.iloc[:,column])
          else:
            df_converted.iloc[:,column] = label_encoders[column].transform(df_converted.iloc[:,column])
    return df_converted, label_encoders


#This function is used for converting model output into categorical form. 
#Used for classification tasks only
def convert_numeric_to_categorical(y, label_encoders):
    df_converted = pd.DataFrame(y)

    # for each column in df
    for column in range(len(df_converted.columns)):
        # Check if the column is numerical
        if pd.api.types.is_numeric_dtype(df_converted.iloc[:,column]):
          df_converted.iloc[:,column] = label_encoders[column].inverse_transform(df_converted.iloc[:,column].astype(int))

    return df_converted



def replace_missing_values_with_neighours_data(data, n_neighbors=1):
    # in this specific dataset, missing values are indicated by '?'
    # so we will need to first replace all '?' values with NaN
    df_replaced = data.replace('?', np.nan)

    # Initialize KNNImputer with the specified number of neighbors
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_numeric = df_replaced.apply(pd.to_numeric, errors='coerce')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
    return df_imputed



# This function can be used for data augmentation
# Below is just one example of data augmentation, but you can use any other suitable data augmentation technique
# If you don't want to use any data augmentation for your task, then in the first line of the function write return X,y 
def augment_data(X,y, noise_factor=0.05):
  """
  Augments the data by adding noise to the 'capital.gain' and 'capital.loss' columns.

  Parameters:  X,y: data before augmentation
              noise_factor: The factor to multiply the standard deviation of the columns by to generate noise.

  Returns:The augmented data
  """
  # Make a copy of the data to avoid modifying the original
  x_noisy = X.copy()

  # Add noise to 'capital.gain' which is column # 10
  noise_gain = np.random.normal(0, noise_factor * X[:,10].std(), size=len(X[:,10]))
  x_noisy[:,10] += noise_gain

  # Add noise to 'capital.loss' which is column # 11
  noise_loss = np.random.normal(0, noise_factor * X[:,11].std(), size=len(X[:,11]))
  x_noisy[:,11] += noise_loss

  #use both original & newly created datapoints for training
  x_augmented = np.vstack([x_noisy, X])
  y_augmented = np.concatenate([y, y])
  return x_augmented, y_augmented



# This function can be used for feature engineering
# Below is just one example of feature engineering where we kept the original raw data and added an extra column. You can use any other features as deemed appropriate for your problem
def feature_engineering(x):
  # todo: add new features as needed
  x_copy = x.copy()
  #example, here we add a new feature: capital.net calculated as the capital.gain-capital.loss
  capital_net = x_copy[:,10] - x_copy[:,11]

  x_copy = np.hstack((x_copy, capital_net.reshape(-1,1)))
  return x_copy

