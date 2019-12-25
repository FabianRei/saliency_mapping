from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def rearrange_pred_labs(preds, labs, train_test_split):
    preds = [list(p) for p in preds]
    labs = [list(l) for l in labs]
    results_pred = []
    results_lab = []
    for s in train_test_split:
        results_pred.append(preds[s].pop(0))
        results_lab.append(labs[s].pop(0))
    results_pred = np.array(results_pred)
    results_lab = np.array(results_lab)
    return results_pred, results_lab


def get_top_subjects(number):
    global subs
    global preds
    global labs
    perc_lab = np.percentile(labs, 100 - number*100/1137)
    subs_lab = subs[labs>perc_lab]
    count_lab = number
    while len(np.unique(subs_lab)) < number:
        count_lab += 1
        perc_lab = np.percentile(labs, 100 - count_lab*100/1137)
        subs_lab = subs[labs>perc_lab]
    perc_pred = np.percentile(preds, 100 - number*100/1137)
    subs_pred = subs[preds>perc_pred]
    count_pred = number
    while len(np.unique(subs_pred)) < number:
        count_pred += 1
        perc_pred = np.percentile(preds, 100 - count_pred*100/1137)
        subs_pred = subs[preds>perc_pred]
    # percentile might get bigger number, if both have the dame delta SUVR pred/lab
    return perc_pred, perc_lab


def get_unique_t0_suvr():
    global t0_suvr
    global subs
    t0, t0_idx = np.unique(subs, return_index=True)
    suvr_unique = t0_suvr[t0_idx]
    return suvr_unique

def get_matches(number):
    global preds
    global labs
    global subs
    perc_pred = np.percentile(preds, 100 - number*100 / 1137)
    perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    # perc_pred, perc_lab = get_top_subjects(76)
    # _, perc_lab = get_top_subjects(77)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (preds > perc_pred)])
    sub_ids_study, study_idx = np.unique(subs[preds>perc_pred], return_index=True)
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (preds > perc_pred)


def get_matches_target(study_size, target_size):
    global preds
    global labs
    global subs
    # perc_pred = np.percentile(preds, 100 - number*100 / 1137)
    # perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    perc_pred, perc_lab = get_top_subjects(study_size)
    _, perc_lab = get_top_subjects(target_size)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (preds > perc_pred)])
    sub_ids_study, study_idx = np.unique(subs[preds>perc_pred], return_index=True)
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (preds > perc_pred)


def get_high_t0suvr_matches(number):
    global preds
    global labs
    global t0_suvr
    high_suvr = np.percentile(t0_suvr, 100 - number*100 / 1137)
    perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    # high_suvr, perc_lab = get_top_subjects(number)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (t0_suvr > high_suvr)])
    sub_ids_study = np.unique(subs[t0_suvr>high_suvr])
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print('matches based on high suvr selection')
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (t0_suvr > high_suvr)

def get_minimal_change_matches(number):
    global preds
    global labs
    perc_low = np.percentile(preds, 50 - number*50/1137)
    perc_high = np.percentile(preds, 50 + number*50.5/1137)
    perc2_low = np.percentile(labs, 50 - number*50/1137)
    perc2_high = np.percentile(labs, 50 + number*50/1137)
    sub_ids = np.unique(subs[(preds > perc_low) & (preds < perc_high) & (labs > perc2_low) & (labs < perc2_high)])
    sub_ids_study = np.unique(subs[(preds > perc_low) & (preds < perc_high)])
    sub_ids_label = np.unique(subs[(labs > perc2_low) & (labs < perc2_high)])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")


def plot_densities():
    global preds
    global labs
    sns.kdeplot(preds, label='GBDT predictions')
    sns.kdeplot(labs, label='Ground truth')
    plt.legend()
    plt.show()


def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5


def rmse_over_time():
    global preds
    global labs
    global delta_time
    delta_time_yrs = delta_time/(365*24*60*60)
    results = []
    for i in range(4):
        start = 1+i*2
        end = 3+i*2
        time_filter = (delta_time_yrs>=start) & (delta_time_yrs < end)
        time_preds = preds[time_filter]
        time_labs = labs[time_filter]
        results.append(rmse(time_preds, time_labs))
    print(results)
    return 0



target_folder = r'C:\Users\Fabian\stanford\gbdt\analysis'
target_file = 'all_results.pickle'
folders = glob(os.path.join(target_folder, '157*'))


folder = folders[-1]
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
y = data['y']
train_test_split = data['train_test_split']
preds, labs = rearrange_pred_labs(preds, labs, train_test_split)
subs = more_data['subs']
t0_suvr = more_data['t0_suvr']
age = more_data['age']
weight = more_data['age']
apoe = more_data['apoe']
faq = more_data['faqtotal']
mmse = more_data['mmsescore']
delta_time = more_data['delta_time']

sort_idxs = np.argsort(preds)[::-1]
sort_idxs2 = np.argsort(labs)[::-1]
print(sort_idxs[:25])
print(labs[sort_idxs][:25])
print(preds[sort_idxs][:25])
perc = np.percentile(preds, 100-2500/1137)
perc2 = np.percentile(labs, 100-2500/1137)
big_preds = preds[preds>perc]
big_labs = labs[labs>perc2]
big_pred_labs = labs[preds>perc]
fis = []
# get_high_t0suvr_matches(116)
for i in range(10, 101, 10):
    print(len(np.unique(subs[get_matches_target(i, 100)]))/i)
get_matches(100)
print(np.sum((labs>perc2) & (preds>perc)))
filter = get_matches(100)
_, sub_filter = np.unique(subs, return_index=True)
rmse_over_time()
for gbm in gbms:
    fis.append(gbm.feature_importance())
fi = fis[0]
for f in fis[1:]:
    fi += f
fi = (fi/5).astype(np.int)

f_names = []
f_names = list(x_names)
for i in range(len(fi)-len(x_names)):
    f_names.append(f'act_{i}')
f_names = np.array(f_names)

print('done')