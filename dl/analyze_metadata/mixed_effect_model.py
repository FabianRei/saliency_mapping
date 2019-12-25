import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
import h5py
import plotly.express as px
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5


def mixed_lm_crossval(data, cval_fold=5):
    train_test = data.pop('test')
    dependent_var = data.pop('delta_suvr')
    apoe = data.pop('apoe')
    for i in np.unique(apoe):
        d_key = f'apoe_{i}'
        d_val = apoe==i
        data[d_key] = d_val
    keys = list(data.keys())
    print(keys)
    rmse_test = []
    rmse_train = []
    for i in range(cval_fold):
        test = train_test == i
        if 'faqtotal' in keys:
            data['faqtotal'][data['faqtotal'] == -1] = np.mean(data['faqtotal'][~test][data['faqtotal'][~test] != -1])
        if 'mmsescore' in keys:
            data['mmsescore'][data['mmsescore'] == -1] = np.mean(data['mmsescore'][~test][data['mmsescore'][~test] != -1])
        independent_var = np.column_stack((data['t0_suvr'], data['sex'], data['weight'], data['delta_time'],
                                           data['mmsescore'], data['faqtotal'], data['age'],
                                           data['apoe_0'], data['apoe_1'], data['apoe_2'], data['apoe_3'], data['apoe_4'], data['apoe_5']))
        dep_test = dependent_var[test]
        indep_test = independent_var[test]
        dep_train = dependent_var[~test]
        indep_train = independent_var[~test]
        model = sm.OLS(dep_train, indep_train)
        results = model.fit()
        dep_test_pred = model.predict(params=results.params, exog=indep_test)
        dep_train_pred = model.predict(params=results.params, exog=indep_train)
        print(rmse(dep_test_pred, dep_test))
        print(rmse(dep_train_pred, dep_train))
        rmse_test.append(rmse(dep_test_pred, dep_test))
        rmse_train.append(rmse(dep_train_pred, dep_train))

        print('db')
    print(f'mean rmse test is {np.mean(rmse_test)}')
    print(f'mean rmse train is {np.mean(rmse_train)}')
    print('db')


data_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
mixed_lm_crossval(data)
# pd_data = pd.DataFrame(data)
#
# md = smf.mixedlm("delta_suvr ~ delta_time + a1_e4 + a2_e4 + weight_meaned + amyloid_status + t0_suvr + sex_f_true", pd_data, groups=pd_data['img_id'])
# mdf = md.fit()
# # print(mdf.summary())
# print('done')
#
# dependent_arr = data['delta_suvr']
# independent_arr = np.column_stack((data['delta_time'], data['a1_e4'], data['a2_e4'], data['t0_suvr'], data['sex_f_true'], data['weight_meaned']))
# independent_arr = sm.add_constant(independent_arr)
# # gamma_model = sm.GLM(dependent_arr, independent_arr, family=sm.families.Gamma())
# # gamma_results = gamma_model.fit()
# # print(gamma_results.summary())
# model = sm.OLS(dependent_arr, independent_arr)
# results = model.fit()
# results.mse_resid
# print(results.summary())
