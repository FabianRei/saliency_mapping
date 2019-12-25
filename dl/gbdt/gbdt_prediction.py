import os, sys, inspect
# this is done to run things from console
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir2)

import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from dl.gbdt.five_fold_cv import cross_validation_gbdt
import multiprocessing as mp


# in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test.pickle'
if os.name == 'nt':
    in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
else:
    in_path = '/share/wandell/data/reith/gbdt/rf_data_train_test_crossval.pickle'
with open(in_path, 'rb') as f:
    data = pickle.load(f)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'max_leaves': 50,
    'max_depth': 9,  # 9 optimal for activations, 4 optimal for metadata only
    'max_bin': 255,
    'num_iterations': 4000,
    'learning_rate': 0.0045,  # 0.0006 optimal for metadata, 0.0045 for activations
    'feature_fraction': 0.5, # 0.8 for metadata only
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'verbose': 0,
    'min_data_in_leaf': 9
}
# for i in range(3, 12):
#     params['max_depth'] = i
#     cross_validation_gbdt(data, params, activations=True)

# params['max_depth'] = 4
#
# # for i in range(5, 30):
# #     params['num_iterations'] = i*100
# #     cross_validation_gbdt(data, params)
#
# params['num_iterations'] = 2000
# # params['learning_rate'] = 0.004
# for i in range(5, 30, 2):
#     params['min_data_in_leaf'] = i
#     cross_validation_gbdt(data, params, activations=False, cval_range=5)
#
# print('done')
with_activations = False

if not with_activations:
    params['learning_rate'] = 0.0006
    params['max_depth'] = 4
    params['feature_fraction'] = 0.8
else:
    params['learning_rate'] = 0.0045
    params['max_depth'] = 9
    params['feature_fraction'] = 0.5
# cross_validation_gbdt(data, params, activations=False, cval_range=5, exclude='sex')
#

cross_validation_gbdt(data, params, activations=with_activations, cval_range=5, extra_folder='exclusions_no_acs')

features = ['t0_suvr', 'sex', 'weight', 'delta_time', 'apoe', 'mmsescore', 'faqtotal', 'age']
for f in features:
    cross_validation_gbdt(data, params, activations=with_activations, cval_range=5, exclude=f, extra_folder='exclusions_no_acs')
# cross_validation_gbdt(data, params, activations=False, cval_range=5)

# new params: 0.009 lr, its 2000, max_depth 9, min_data_in_leaf 9
params['learning_rate'] = 0.0006
params['num_iterations'] = 4000
params['min_data_in_leaf'] = 9
params['max_depth'] = 4 # 11 also quite good
# cross_validation_gbdt(data, params, activations=True, cval_range=5)
# sub_processes = []
# for i in range(3, 21, 3):
#     params['max_depth'] = i
#     print(params)
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='depth_search_cv')

# params['max_depth'] = 9
# params['min_data_in_leaf'] = 9
# # params['boosting_type'] = 'dart'
# for i in range(2, 45, 2):
#     params['learning_rate'] = 0.0001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='lr_9_9')
#
# params['max_depth'] = 11
# params['min_data_in_leaf'] = 9
# # params['boosting_type'] = 'dart'
# for i in range(5, 66, 5):
#     params['learning_rate'] = 0.0001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='lr_11_9')
#
# params['max_depth'] = 11
# params['min_data_in_leaf'] = 13
# # params['boosting_type'] = 'dart'
# for i in range(5, 66, 5):
#     params['learning_rate'] = 0.0001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='lr_11_13')
#
# params['max_depth'] = 9
# params['min_data_in_leaf'] = 13
# # params['boosting_type'] = 'dart'
# for i in range(2, 60, 2):
#     params['learning_rate'] = 0.0001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='lr_9_13')

# for i in range(3, 18, 2):
#     params['min_data_in_leaf'] = i
#     print(params)
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='leaf_search_cv')
    # spreading out into different processes does not seem to speed up GBDT prediction
    # params['learning_rate'] = i*0.0045
    # sub_proc = mp.Process(target=cross_validation_gbdt, args=[data, params], kwargs={'activations': True,
    #                                                                                  'cval_range': 5,
    #                                                                                  'extra_folder': 'lr_search'})
    #
    # sub_proc.start()
    # sub_processes.append(sub_proc)
    # if len(sub_processes)<10:
    #     continue
    # while len(sub_processes) >= 10:
    #     for j, sub in enumerate(sub_processes):
    #         if not sub.is_alive():
    #             sub_processes.pop(j)
    #             break

#cross_validation_gbdt(data, params, activations=True, cval_range=5, extra_folder='lr_search')

# params['max_depth'] = 11
#
# for i in range(6, 21, 3):
#     params['min_data_in_leaf'] = i
#     cross_validation_gbdt(data, params, activations=True, cval_range=1)
#
# params['min_data_in_leaf'] = 9
#
# for i in range(1, 12, 2):
#     params['learning_rate'] = 0.001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=1)


# for feat in features:
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, exclude=feat)
