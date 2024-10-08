
import numpy as np
import pandas as pd
import pickle
import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# if mode is 'develop', it means that we will use the available data for training our model
#if mode is 'evaluate', it means that we will load the model that was trained before (as well as any operators for data pre-processing) to make prediction on novel data
mode = 'develop'

#How many outputs (i.e., columns) your model should predict
num_outputs  = 1

if mode == 'develop':
  #read data from file
  all_data = pd.read_csv('./data/adult1.csv')
  print("reading data complete")

  #convert categorical data to numeric (similar to what we did in Lecture 4 - page 10)
  label_encoders = []
  all_data_numeric,label_encoders = Preprocessing.convert_categorical_to_numeric(all_data, label_encoders, mode)
  print("categorical to numerical conversion complete")
  print("shape of data before categorical concersion", all_data.shape)
  print("shape of data after categorical concersion", all_data_numeric.shape)


  #replace missing data with data from the most similar raw (similar to what we did in Lecture 4 - page 6)
  all_data_numeric_no_missing = Preprocessing.replace_missing_values_with_neighours_data(all_data_numeric)
  print("missing data replacement complete")

  
  X = all_data_numeric_no_missing.iloc[:, :-num_outputs] # inputs 
  y = all_data_numeric_no_missing.iloc[:, -num_outputs:] # outputs
  # split into train & test sets
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

  # Standardise the traing data (similar to what we did in Lecture 4 - pages 15-16)
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)


  #augment data (similar to what we did in Lecture 4 - pages 39)
  x_train_augmented, y_train_augmented = Preprocessing.augment_data(x_train_scaled,y_train)

  # Apply feature engineering 
  x_train_feature_engineered = Preprocessing.feature_engineering(x_train_augmented)

  # Apply PCA to reduce the dimensionality of the data (similar to what we did in Lecture 4 - pages 28-29)
  pca = PCA(n_components=10)
  pca.fit(x_train_feature_engineered)
  x_train_pca = pca.transform(x_train_feature_engineered)

  #Train the model
  # we specify that 10% of the training data will be used for validation. Ealy stopping is applied (similar to what we did in Lecture 4, page 33)
  # we also use alpha =0.1 to activate L2 regularisation
  model = MLPClassifier(hidden_layer_sizes=(100, 50), activation= 'relu', early_stopping = True, validation_fraction =0.1,alpha =0.1 , verbose = True )
  model.fit(x_train_pca, y_train_augmented)
  train_score = model.score(x_train_pca, y_train_augmented)
  print("Training score:", train_score)

  # test model
  x_test_scaled = scaler.transform(x_test)
  x_test_feature_engineered = Preprocessing.feature_engineering(x_test_scaled)
  x_test_pca = pca.transform(x_test_feature_engineered)
  test_score = model.score(x_test_pca, y_test)
  print("Testing score:", test_score)

  # now save the model, the PCA object, the scaler object & the encoders so that you can load them at a later point & use them for novel data
  with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

  with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

  with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

  with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)



# Evaluate the model
elif mode == 'evaluate':
  # read the pre-trained model, the PCA object & the scaler object & encoders
  model = pickle.load(open('model.pkl', 'rb'))
  pca = pickle.load(open('pca.pkl', 'rb'))
  scaler = pickle.load(open('scaler.pkl', 'rb'))
  label_encoders= pickle.load(open('label_encoders.pkl', 'rb'))

  # Read novel data. This data usually has no output y. You'll need to use your model to predict y
  novel_data= pd.read_csv('./data/adult2.csv')
  
  # you will need to do the exact same pre-preocessing for this novel data similar to what you did for your training data

  #convert categorical data to numeric
  novel_data_numeric, _ = Preprocessing.convert_categorical_to_numeric(novel_data, label_encoders, mode)

  #replace missing data with data from the most similar raw
  novel_data_no_missing = Preprocessing.replace_missing_values_with_neighours_data(novel_data_numeric)



  X_novel = novel_data_no_missing    # novel data will not contain any output; so use all columns as X
  X_novel = scaler.transform(X_novel)
  # standardise 
  X_novel = Preprocessing.feature_engineering(X_novel)
  # get pca components
  X_novel_pca = pca.transform(X_novel)
  y_predict = model.predict(X_novel_pca)
  print("prediction complete")
  #if output was originally categorical, then convert back to its original form
  y_predict = Preprocessing.convert_numeric_to_categorical(y_predict, label_encoders[-num_outputs:])
  print(y_predict)
  #save the prediction to a csv file
  pd.DataFrame(y_predict).to_csv('prediction.csv')