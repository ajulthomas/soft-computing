# Assignment 1

## Soft Computing PG (7197)

**University of Canberra**

Ajul Thomas (_u3253992@uni.canberra.edu.au_)
30 August 2024

### Parameter combinations used for the experiment

```py
default_params

params_one = { "random_state": 33, "activation": "relu", "hidden_layer_sizes": (12, 24, 48, 96), "learning_rate_init": 0.001, "batch_size": 20 }

params_two = { "random_state": 33, "activation": "relu", "hidden_layer_sizes": (12, 24, 48, 96), "learning_rate_init": 0.001, "batch_size": 50 }

```

### Other Model 1

This model was intended to test the performance of the model on the given dataset without applying any feature engineering or transformation techniques. This is to understand the impact of some of the hyper params on the basic model.

**Features used:** All features of the given data set were used.
**Data split:** 70% of data is used for training, with 10% of validation and 20% for testing.
**Evaluation metrics:** RMSE and MAE scores were used to evaluate the performance of the model.
**Observations:**

| **Model Params** | **Train** **(rmse)** | **Validate** **(rmse)** | **Test** **(rmse)**  | **Evaluate** **(rmse)** |
| :--------------: | :------------------: | :---------------------: | :------------------: | :---------------------: |
|  default_params  | 0.010235990698238237 |   0.01048744262284605   | 0.009904215868987822 |  0.010028905979234757   |
|    params_one    | 0.006932223912952257 |  0.006729120103675562   | 0.006602403372255023 |  0.006681047017901092   |
|    params_two    | 0.007888354736637352 |  0.007768264985211542   | 0.007381068902074671 |  0.007521259696380296   |

### Other Model 2

This model was intended to test the performance of the model, after feature engineering and standardisation.

**Features used:** The candle positions and robot positions were used to calculate the euclidean distance between each candle and robot, which were used along with collision and orientation of robot.  
**Data split:** 70% of data is used for training, with 10% of validation and 20% for testing.
**Evaluation metrics:** RMSE and MAE scores were used to evaluate the performance of the model.
**Observations:**

### Other Model 3

This model evaluates the model after addition of more features and uses standardisation and transformation techniques.

**Features used:** The candle positions and robot positions were used to calculate the distance of each robot with the candles along the x and y axis seperately, which were used along with collision and orientation of robot.

**Data split:** 70% of data is used for training, with 10% of validation and 20% for testing.

**Evaluation metrics:** RMSE and MAE scores were used to evaluate the performance of the model.

**Observations:**

### Final Model

This model evaluates the model after addition of more features and uses standardisation and transformation techniques.

**Features used:**

**Data split:** 70% of data is used for training, with 10% of validation and 20% for testing.

**Evaluation metrics:** RMSE and MAE scores were used to evaluate the performance of the model.

**Observations:**
