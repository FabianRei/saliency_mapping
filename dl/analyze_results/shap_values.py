from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import shap


def get_xy():
    global data
    x = data['x']
    y = data['y']
    xx = []
    yy = []
    tts = data['train_test_split']
    for split in np.unique(tts):
        x_vals = x[tts == split]
        xx.append(x_vals)
        y_vals = y[tts==split]
        yy.append(y_vals)
    return xx, yy


def get_shap_names():
    global x_values
    global x_names
    num_features = x_values[0].shape[-1]
    shap_names = x_names.copy()
    for i in range(num_features - len(x_names)):
        shap_names.append(f'activation_{i+1}')
    shap_names = np.array(shap_names)
    return shap_names


def get_shap_values():
    global preds
    global labs
    global gbms
    global x_values
    all_shap = []
    shap_names = get_shap_names()
    for l, g, x in zip(labs, gbms, x_values):
        shap_values = shap.TreeExplainer(g).shap_values(x, l)
        all_shap.extend(shap_values)
    all_shap = np.array(all_shap)
    all_x = np.concatenate(x_values)
    shap.summary_plot(all_shap, features=all_x, feature_names=shap_names, max_display=8)
    print('nice')


target_folder = r'C:\Users\Fabian\stanford\gbdt\analysis'
target_file = 'all_results.pickle'
folders = glob(os.path.join(target_folder, '157*'))


folder = folders[-2]
target = os.path.join(folder, target_file)
with open(target, 'rb') as f:
    data = pickle.load(f)

in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
with open(in_path, 'rb') as f:
    more_data = pickle.load(f)

gbms = data['gbm']
x_names = data['x_names']
preds = data['predictions']
labs = data['labels']
x_values, y_values= get_xy()

get_shap_values()

print('done')