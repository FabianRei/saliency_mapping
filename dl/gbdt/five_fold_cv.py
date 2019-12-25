import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pandas as pd
import os
import time
import pprint


def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5


def get_xy(data, activations, exclude):
    x = []
    x_names = []
    for k, v in data.items():
        if k == 'delta_suvr':
            y = v
        elif k != 'activations' and k != 'delta_suvr' and k != 'suvr' and k != 'test' and k != 'subs' and k != exclude:
            x_names.append(k)
            x.append(v)
        # elif k == 'activations':
        #     x.extend(list(v.T))
        # elif k == 'delta_time':
        #     x.append(v)
        # elif k == 't0_suvr':
        #     x.append(v)

    # idx = x_names.index('sex')
    # x[idx] = x[idx].astype(np.int)
    # idx = x_names.index('apoe')
    # x[idx] = x[idx]
    # add activations:
    if activations:
        x.extend(list(data['activations'].T))
    x = np.array(x).T
    train_test_split = data['test']
    # x = pd.DataFrame(x)
    return x, y, train_test_split, x_names


def save_results(res, activations, exclude='', extra_folder=''):
    global out_path
    output_path = os.path.join(out_path, extra_folder)
    if activations:
        acs = 'w_acs_'
    else:
        acs = 'wo_acs_'
    if exclude != '':
        excluded = f'_{exclude}_excluded'
    else:
        excluded = ''
    folder_name = f'{int(time.time())}_{acs}_{str(np.mean(res["rmse"])).replace(".", "_")}{excluded}'
    out_dir = os.path.join(output_path, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    res_csv = {'RMSE': res['rmse'], 'RMSE_Y_MEAN': res['rmse_mean'], 'RMSE_TRAIN': res['rmse_train'],
               'RMSE_Y_MEAN_TRAIN': res['rmse_mean_train']}
    res_csv_mean = {}
    for k, v in res_csv.items():
        res_csv[k] = np.array(v)
        res_csv_mean[k] = np.array([np.mean(v)])
    res_csv = pd.DataFrame.from_dict(res_csv)
    res_csv_mean = pd.DataFrame.from_dict(res_csv_mean)
    print(res_csv_mean)
    res_csv.to_csv(os.path.join(out_dir, 'results.csv'))
    res_csv_mean.to_csv(os.path.join(out_dir, 'results_mean.csv'))
    with open(os.path.join(out_dir, 'params.txt'), 'w') as f:
        pprint.pprint(res['params'], f)
    with open(os.path.join(out_dir, 'all_results.pickle'), 'wb') as f:
        pickle.dump(res, f)
    res['x'] = None
    with open(os.path.join(out_dir, 'some_results.p'), 'wb') as f:
        pickle.dump(res, f)
    print('nice')


def cross_validation_gbdt(data, params, activations=False, cval_range=5, exclude='', extra_folder=''):
    params['verbose'] = -1
    print(params)
    x, y, train_test_split, x_names = get_xy(data, activations, exclude=exclude)
    print(x_names)
    res = {'predictions': [], 'labels': [], 'rmse': [], 'pred_train': [], 'labels_train': [],
           'rmse_train': [], 'rmse_mean': [], 'rmse_mean_train': [], 'gbm': []}
    # params['categorical_feature'] = [1, 4, 5, 6]
    for split in range(cval_range):
        test = train_test_split == split
        # use train mean val for not known faqtotal and mmse
        try:
            faq_index = x_names.index('faqtotal')
            x[:, faq_index][x[:, faq_index]==-1] = np.mean(x[:, faq_index][~test][x[:, faq_index][~test]!=-1])
        except:
            pass
        try:
            mmse_index = x_names.index('mmsescore')
            x[:, mmse_index][x[:, mmse_index] == -1] = np.mean(x[:, mmse_index][~test][x[:, mmse_index][~test] != -1])
        except:
            pass

        x_train = x[~test]
        y_train = y[~test]
        d_train = lgb.Dataset(x_train, label=y_train)
        x_test = x[test]
        y_test = y[test]
        warnings.simplefilter('ignore', UserWarning)
        gbm = lgb.train(params, d_train)
        warnings.simplefilter('default', UserWarning)
        y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
        y_pred_train = gbm.predict(x_train, num_iteration=gbm.best_iteration)
        res['predictions'].append(y_pred)
        res['labels'].append(y_test)
        res['rmse'].append(rmse(y_test, y_pred))
        res['rmse_train'].append(rmse(y_pred_train, y_train))
        res['pred_train'].append(y_pred_train)
        res['labels_train'].append(y_train)
        res['rmse_mean'].append(rmse(y_test, np.ones(y_test.shape)*y_train.mean()))
        res['rmse_mean_train'].append(rmse(y_train, np.ones(y_train.shape) * y_train.mean()))
        res['gbm'].append(gbm)
        print(rmse(y_test, y_pred), rmse(y_test, np.ones(y_test.shape)*y_train.mean()))
        print(rmse(y_train, y_pred_train), rmse(y_train, np.ones(y_train.shape) * y_train.mean()))
    res['params'] = params
    res['x_names'] = x_names
    res['x'] = x
    res['y'] = y
    res['train_test_split'] = train_test_split
    save_results(res, activations, exclude=exclude, extra_folder=extra_folder)
    return res

if os.name == 'nt':
    out_path = r'C:\Users\Fabian\stanford\gbdt'
elif os.name == 'posix':
    out_path = r'/share/wandell/data/reith/gbdt'