import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from dl.gbdt.five_fold_cv import cross_validation_gbdt

# in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test.pickle'
in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
with open(in_path, 'rb') as f:
    data = pickle.load(f)

x = []
x_name = []
for k, v in data.items():
    if k == 'delta_suvr':
        y = v
    elif k != 'activations' and k != 'delta_suvr' and k != 'suvr' and k != 'test' and k != 'subs':
        x_name.append(k)
        x.append(v)
    # elif k == 'activations':
    #     x.extend(list(v.T))
    # elif k == 'delta_time':
    #     x.append(v)
    # elif k == 't0_suvr':
    #     x.append(v)

# add activations:
# x.extend(list(data['activations'].T))
x = np.array(x).T
test = data['test']
# test = test == 0
x_train = x[~test]
y_train = y[~test]
categories = [1, 4, 5, 6]

d_train = lgb.Dataset(x_train, label=y_train)
x_test = x[test]
y_test = y[test]
d_test = lgb.Dataset(x_test, label=y_test)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'max_leaves': 44,
    'max_depth': 8,
    'max_bin': 255,
    'num_iterations': 1000,
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'verbose': 0,
}
cross_validation_gbdt(data, params)


print('Starting training...')
# train
gbm = lgb.train(params,
                d_train)

print('Saving model...')
# save model to file
# gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
y_pred_train = gbm.predict(x_train, num_iteration=gbm.best_iteration)

# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('RMSE mean prediction is:', mean_squared_error(y_test, np.ones(y_test.shape)*np.mean(y_train.mean())) ** 0.5)
print('RMSE training is:',  mean_squared_error(y_train, y_pred_train) ** 0.5)
print('RMSE mean prediction is:', mean_squared_error(y_train, np.ones(y_train.shape)*np.mean(y_train.mean())) ** 0.5)

print('done')