import pandas as pd
import pickle
import numpy as np
import os
import re


# get dict val
def gdv(data, val, keys, mask=None, substract_min=False, set_min=False, grouping=None, min_reference=None):
    res = []
    for k in keys:
        res.append(data[k][val])
    if isinstance(res[0], pd.core.series.Series):
        res = [float(r) for r in res]
    res_types = [type(r) for r in res]
    if str in res_types:
        value_error = False
        try:
            [float(r) for r in res]
        except Exception as e:
            value_error = type(e)==ValueError
        if not value_error:
            res = [float(r) for r in res]
    res = np.array(res)
    if mask is not None:
        res = res[mask]
    if substract_min:
        res = subtract_minimum(res, grouping, min_reference)
    if set_min:
        res = set_minimum(res, grouping, min_reference)
    return res


def set_minimum(res, grouping, min_reference):
    for g in np.unique(grouping):
        filter = grouping==g
        res[filter] = res[filter][np.argmin(min_reference[filter])]
    return res


def subtract_minimum(res, grouping, min_reference):
    for g in np.unique(grouping):
        filter = grouping == g
        res[filter] -= res[filter][np.argmin(min_reference[filter])]
    return res


def get_id(name):
    return re.search(r'I\d{3,10}', name).group()


fp_pickle = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal_times_fixed.pickle'
fp = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\feature_activations_fixed\fc_activations.pickle'

with open(fp_pickle, 'rb') as f:
    labels = pickle.load(f)
with open(fp, 'rb') as f:
    activations = pickle.load(f)



k_act = list(activations.keys())
k_lab = list(labels.keys())
k_lab = [k for k in k_lab if get_id(k) in k_act]
k_lab = sorted(k_lab, key=lambda x: get_id(x))
k_act = sorted(k_act)


print(activations[k_act[0]])
print(labels[k_lab[0]])
train = gdv(labels, 'train_data', k_lab)
test = ~train
# res_acts = gdv(activations, 'activations', k_act, mask=test)
# res_acts = np.squeeze(res_acts, 1)
sub_ids = gdv(labels, 'rid', k_lab, mask=test)
examdate_posix = gdv(labels, 'examdate_posix', k_lab, mask=test)
suvrs = gdv(labels, 'label_0_79_suvr', k_lab, mask=test)


delta_suvrs = gdv(labels, 'label_0_79_suvr', k_lab, mask=test, substract_min=True, grouping=sub_ids, min_reference=examdate_posix)
weight = gdv(labels, 'weight', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
delta_time = gdv(labels, 'examdate_posix', k_lab, mask=test, substract_min=True, grouping=sub_ids, min_reference=examdate_posix)
sex = gdv(labels, 'sex', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
sex = sex == 'F'
faqtotal = gdv(labels, 'faqtotal', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
# faqtotal[faqtotal==-1] = np.mean(faqtotal[faqtotal!=-1])
age = gdv(labels, 'age', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
res_acts = gdv(activations, 'activations', k_act, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
res_acts = np.squeeze(res_acts, 1)

apoea1 = gdv(labels, 'apoea1', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
apoea2 = gdv(labels, 'apoea2', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)

apoe = []
for a1, a2 in zip(apoea1, apoea2):
    if a1 == 2 and a2 == 2:
        apoe.append(0)
    elif a1 == 2 and a2 == 3:
        apoe.append(1)
    elif a1 == 3 and a2 == 3:
        apoe.append(2)
    elif a1 == 2 and a2 == 4:
        apoe.append(3)
    elif a1 == 3 and a2 == 4:
        apoe.append(4)
    elif a1 == 4 and a2 == 4:
        apoe.append(5)

apoe = np.array(apoe)
mmsescore = gdv(labels, 'mmsescore', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
# mmsescore[mmsescore==-1] = np.mean(mmsescore[mmsescore!=-1])
suvr_t0 = gdv(labels, 'label_0_79_suvr', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
img_id = gdv(labels, 'img_id', k_lab, mask=test, set_min=True, grouping=sub_ids, min_reference=examdate_posix)
amyloid_status = suvr_t0 > 0.79

res_dict = {'t0_suvr': suvr_t0, 'delta_suvr': delta_suvrs, 'sex': sex, 'weight': weight, 'suvr': suvrs, 'delta_time': delta_time,
            'apoe': apoe, 'activations': res_acts, 'mmsescore': mmsescore, 'faqtotal': faqtotal, 'subs': sub_ids, 'age': age}

np.random.seed(42)
shuff_idxs = np.random.permutation(len(apoe))
for k, v in res_dict.items():
    res_dict[k] = v[shuff_idxs]
t0_filter = res_dict['delta_time']>0
for k, v in res_dict.items():
    res_dict[k] = v[t0_filter]
test = np.zeros(res_dict[k].shape, np.bool)-1

# train test split 0.2 - do five times to extend to five fold cross validation.
for s in np.unique(res_dict['subs']):
    samples = res_dict['subs']==s
    if len(test)*0.2>np.sum(test==0):
        test[samples] = 0
    elif len(test) * 0.2 > np.sum(test == 1):
        test[samples] = 1
    elif len(test)*0.2>np.sum(test==2):
        test[samples] = 2
    elif len(test)*0.2>np.sum(test==3):
        test[samples] = 3
    if len(test)*0.2>np.sum(test==4):
        test[samples] = 4
out_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
res_dict['test'] = test
with open(out_path, 'wb') as f:
    pickle.dump(res_dict, f)
print('nice')
