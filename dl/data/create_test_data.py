import numpy as np
import pickle
from glob import glob
import os
import csv
import h5py
from dl.data.get_dataset import get_dataset
import scipy.misc
from scipy import ndimage

class CsvWriter:
    def __init__(self, file_path, header, default_vals=None, lock=None, delete_if_exists=False):
        self.lock = lock
        self.fp = file_path
        self.header = header
        self.write_header()
        self.default_vals = default_vals
        if delete_if_exists:
            if os.path.isfile(self.fp):
                os.remove(self.fp)

    def write_header(self):
        if self.lock is not None:
            self.lock.acquire()
        if os.path.isfile(self.fp):
            if self.lock is not None:
                self.lock.release()
            return
        with open(self.fp, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=self.header, restval=-1)
            writer.writeheader()
        if self.lock is not None:
            self.lock.release()

    def write_row(self, **kwargs):
        row_dict = {}
        keys = []
        for key, val in kwargs.items():
            row_dict[key] = val
            keys.append(key)
        for h in self.header:
            if h not in keys:
                row_dict[h] = self.default_vals[h]
        if self.lock is not None:
            self.lock.acquire()
        with open(self.fp, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=self.header, restval=-1)
            writer.writerow(row_dict)
        if self.lock is not None:
            self.lock.release()

def get_contrast(path):
    part2 = path.split('_')[-8]
    part1 = path.split('_')[-9]
    contrast = part1 + '.' + part2
    contrast = float(contrast)
    return contrast


def sort_paths(paths):
    paths.sort(key=lambda x: int(x.split('_')[-8]))
    return paths


def write_csv_row(resultCSV, testAcc, accOptimal, d1, d2, dataContrast, nn_dprime):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index',
                   'contrast', 'nn_dprime']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1,
                         'optimal_observer_d_index': d2, 'contrast': dataContrast,
                         'nn_dprime': nn_dprime})


def write_csv_svm(resultCSV, svm_accuracy, dprime_accuracy, contrast, samples_used=1000):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['svm_accuracy', 'dprime_accuracy', 'contrast', 'samples_used']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(
            {'svm_accuracy': svm_accuracy, 'dprime_accuracy': dprime_accuracy, 'contrast': contrast, 'samples_used': samples_used})


def calculate_values(oo_data, nn_data, svm_data, out_path, adjust_imbalance=True):
    nn_data = sort_paths(nn_data)
    oo_data = sort_paths(oo_data)
    svm_data = sort_paths(svm_data)
    csv_path = os.path.join(out_path, 'results.csv')
    if os.path.exists(csv_path):
        os.replace(csv_path, os.path.join(out_path, 'results_old.csv'))
    csv_svm = os.path.join(out_path, 'svm_results_seeded.csv')
    if os.path.exists(csv_svm):
        os.replace(csv_svm, os.path.join(out_path, 'svm_results_seeded_old.csv'))
    for o, n, s in zip(oo_data, nn_data, svm_data):
        with open(o, 'rb') as f:
            oo = pickle.load(f)
        with open(n, 'rb') as f:
            nn = pickle.load(f)
        with open(s, 'rb') as f:
            svm = pickle.load(f)
        if not (get_contrast(o) == get_contrast(n) == get_contrast(s)):
            print('ERROR, contrast not the same..')
            raise AttributeError
        contrast = get_contrast(o)
        oo_acc = np.mean(oo[:,0] == oo[:,1])
        nn_acc = np.mean(nn[:,0] == nn[:,1])
        oo_dprime = calculate_dprime(oo, adjust_imbalance=adjust_imbalance)
        nn_dprime = calculate_dprime(nn, adjust_imbalance=adjust_imbalance)
        svm_dprime = calculate_dprime(svm, adjust_imbalance=adjust_imbalance)
        svm_acc = np.mean(svm[:,0] == svm[:,1])
        write_csv_row(csv_path, nn_acc, oo_acc, -1, oo_dprime, contrast, nn_dprime)
        write_csv_svm(csv_svm, svm_acc, svm_dprime, contrast)


# create result csv files based on pickle prediction-labels
fp1 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed\seed_11\one_slice\epoch_29_pred_labels_train_test_epoch_09-23_17-50_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10.p'
with open(fp1, 'rb') as f:
    res1 = pickle.load(f)
fph5 = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\trial_sample\slice_data_subj.h5'
label_names = ['label_suvr', 'label_amyloid']
data, labels, labels2, s_ids = get_dataset(fph5, label_names=label_names, include_subjects=True)
test = res1['test']
train = res1['train']
np.random.seed(33)
perm_test = np.random.permutation(len(test))
t = test>1.11
l = []
l.extend(test[:,1])
l.extend(train[:,1])
n, c = np.unique(l, return_counts=True)
double = n[c>1]
fp2 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed\seed_11\one_slice\subject_split_09-23_17-50_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10.p'
with open(fp2, 'rb') as f:
    split = pickle.load(f)

data = data[split['shuffle_permutation']]
labels = labels[split['shuffle_permutation']]
l_test = labels[split['test_idxs']]
d_test = data[split['test_idxs']]
stest = test[perm_test]
sl_test = l_test[perm_test]
sd_test = d_test[perm_test]
st = stest>1.11
out_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\human_inspection'
path_labeled = os.path.join(out_path, 'mistakes_labeled')
path_guess = os.path.join(out_path, 'mistakes_guess')
os.makedirs(path_labeled, exist_ok=True)
os.makedirs(path_guess, exist_ok=True)
csv_path = os.path.join(out_path, 'labels_mistakes.csv')
writer = CsvWriter(csv_path, ['i', 'prediction', 'label', 'prediction_amyloid', 'label_amyloid', 'correct', 'squared_error'])

# pp = stest[:,0]
# ll = stest[:,1]
# diff = np.abs(pp-ll)
# sort_idxs = np.argsort(-diff)
# sd_test = sd_test[sort_idxs]
# stest = stest[sort_idxs]

ppp = st[:,0]
lll = st[:,1]
mistakes = ppp != lll
sd_test = sd_test[mistakes]
stest = stest[mistakes]
# for i, dat, pred, label in zip(range(100), sd_test[100:200], stest[100:200, 0], stest[100:200, 1]):
for i, dat, pred, label in zip(range(100), sd_test[:100], stest[:100, 0], stest[:100, 1]):
    f_guess = f'image_{i}.jpg'
    f_label = f'image_{i}_pred_{pred}_label_{label}.jpg'
    dat = ndimage.rotate(dat, 90)
    scipy.misc.imsave(os.path.join(path_guess, f_guess), dat)
    scipy.misc.imsave(os.path.join(path_labeled, f_label), dat)
    a_pred = int(pred>1.11)
    a_label = int(label>1.11)
    squared_error = (pred-label)**2
    writer.write_row(i=i, prediction=pred, label=label, prediction_amyloid=a_pred, label_amyloid=a_label, correct=int(a_pred==a_label), squared_error=squared_error)


    print(i, pred,label)


print(np.mean(st[100:200,0] == st[100:200,1]))


print('nice')
