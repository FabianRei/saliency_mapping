import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
import h5py
import plotly.express as px
import statsmodels



def get_key(key_ids, id):
    for k in key_ids:
        if id == k.split('_')[-1]:
            return k


def get_id(ids, key):
    for i in ids:
        if i == key.spit('_')[-1]:
            return i


# label_path = '/share/wandell/data/reith/federated_learning/labels_detailled.pickle'
h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data_longitudinal_fixed.h5'

data = h5py.File(h5_path, 'r')


in_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal_times_fixed.pickle'
in_path_csf = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal_csf.pickle'
with open(in_path, 'rb') as f:
    data_pickle = pickle.load(f)

with open(in_path_csf, 'rb') as f:
    data_pickle_csf = pickle.load(f)

ages = []
apoea1 = []
apoea2 = []
dead = []
train_data = []
scan_time = []
sub_id = []
faq_total = []
img_id = []
weight = []
sex = []
label_suvr = []
label_amyloid = []
mmsescore = []
composite_suvr = []
csf_suvr = []
exam_dates = []
# scan_keys = np.array(scan_sheet['Scanner'])
# scanner_keys = np.array([','.join(f.split(',')[:-1]) for f in scan_keys])
# scan_numbers = np.array(scan_sheet['Type'])
for k in data.keys():
    ages.append(data[k].attrs['age'])
    apoea1.append(data[k].attrs['apoea1'])
    apoea2.append(data[k].attrs['apoea2'])
    dead.append(data[k].attrs['dead'])
    train_data.append(data[k].attrs['train_data'])
    scan_time.append(data_pickle[k]['scan_time'])
    sub_id.append(data[k].attrs['rid'])
    faq_total.append(data[k].attrs['faqtotal'])
    img_id.append(data[k].attrs['img_id'])
    weight.append(data[k].attrs['weight'])
    sex.append(data[k].attrs['sex'])
    label_suvr.append(data[k].attrs['label_suvr'])
    label_amyloid.append(data[k].attrs['label_amyloid'])
    mmsescore.append(data[k].attrs['mmsescore'])
    composite_suvr.append(data[k].attrs['label_0_79_suvr'])
    csf_suvr.append(data_pickle_csf[k]['label_csf_suvr'])
    exam_dates.append(data[k].attrs['examdate_posix'])


ages = np.array(ages).astype(np.float)
apoea1 = np.array(apoea1).astype(np.float)
apoea2 = np.array(apoea2).astype(np.float)
train_data = np.array(train_data)
scan_time = np.array(scan_time).astype(np.float)
sub_id = np.array(sub_id)
faq_total = np.array(faq_total).astype(np.float)
img_id = np.array(img_id)
weight = np.array(weight).astype(np.float)
sex = np.array(sex)
label_suvr = np.array(label_suvr)
label_amyloid = np.array(label_amyloid)
mmsescore = np.array(mmsescore)
composite_suvr = np.array(composite_suvr)
csf_suvr = np.array(csf_suvr)
exam_dates = np.array(exam_dates)

suvr = label_suvr[~train_data]
suvr_comp = composite_suvr[~train_data]
suvr_csf = csf_suvr[~train_data]
time = scan_time[~train_data]
sub = sub_id[~train_data]
d_apoea1 = apoea1[~train_data]
d_apoea2 = apoea2[~train_data]
d_ages = ages[~train_data]
d_weight = weight[~train_data]
d_sex = sex[~train_data]
d_img_id = img_id[~train_data]
d_exam_dates = exam_dates[~train_data]

delta_s = []
delta_s_comp = []
delta_s_csf = []
delta_t = []
d_suvrs = []
d_suvrs_comp = []
d_suvrs_csf = []
apoe1 = []
apoe2 = []
age = []
weight_te = []
sex_te = []
sub_te = []
img_id_te = []
exam_dates_te = []

for s in np.unique(sub):
    su = suvr[sub==s]
    su_comp = suvr_comp[sub==s]
    su_csf = suvr_csf[sub==s]
    ti = time[sub==s]
    a1 = d_apoea1[sub==s]
    a2 = d_apoea2[sub==s]
    ag = d_ages[sub==s]
    we = d_weight[sub==s]
    se = d_sex[sub==s]
    imid = d_img_id[sub==s]
    ex = d_exam_dates[sub==s]
    if ti.min() >0:
        if len(ti) > 1:
            ti -= ti.min()
        else:
            continue
    delta_su = su-su[ti.argmin()]
    delta_su_comp = su_comp-su_comp[ti.argmin()]
    delta_su_csf = su_csf-su_csf[ti.argmin()]
    if delta_su.min() < -0.2:
        print('db')
    delta_s.extend(delta_su)
    delta_s_comp.extend(delta_su_comp)
    delta_s_csf.extend(delta_su_csf)
    delta_t.extend(ti)
    d_suvrs.extend(su)
    d_suvrs_comp.extend(su_comp)
    d_suvrs_csf.extend(su_csf)
    apoe1.extend(a1)
    apoe2.extend(a2)
    age.extend(ag)
    weight_te.extend(we)
    sex_te.extend(se)
    sub_te.extend(sub[sub==s])
    img_id_te.extend(imid)
    exam_dates_te.extend(ex)

delta_s = np.array(delta_s)
delta_s_comp = np.array(delta_s_comp)
delta_s_csf = np.array(delta_s_csf)
delta_t = np.array(delta_t)
d_suvrs = np.array(d_suvrs)
d_suvrs_comp = np.array(d_suvrs_comp)
d_suvrs_csf = np.array(d_suvrs_csf)
apoe1 = np.array(apoe1)
apoe2 = np.array(apoe2)
age = np.array(age)
weight_te = np.array(weight_te)
exam_dates_te = np.array(exam_dates_te)
sex_te = np.array(sex_te)
t_sum = np.array(d_suvrs)>1.11
t0_suvr = delta_s + d_suvrs
t0_suvr_comp = delta_s_comp + d_suvrs_comp
t0_suvr_csf = delta_s_csf + d_suvrs_csf
sub_te = np.array(sub_te)
img_id_te = np.array(img_id_te)


weight_te_meaned = weight_te.copy()
weight_te_meaned[weight_te_meaned==0] = weight_te_meaned[weight_te_meaned>0].mean()

a1_e2 = (apoe1==2).astype(np.int)
a1_e3 = (apoe1==3).astype(np.int)
a1_e4 = (apoe1==4).astype(np.int)

a2_e2 = (apoe2==2).astype(np.int)
a2_e3 = (apoe2==3).astype(np.int)
a2_e4 = (apoe2==4).astype(np.int)
t_sum = t_sum.astype(np.int)
sex_f_true = (sex_te=='F').astype(np.int)

meta_data = {'delta_suvr': delta_s, 't0_suvr': t0_suvr, 'apoea1': apoe1, 'apoea2': apoe2, 'weight': weight_te,
             'weight_meaned': weight_te_meaned, 'sex': sex_te, 'delta_time': delta_t, 'suvrs': d_suvrs,
             'age': age, 'amyloid_status': t_sum, 'sub_id': sub_te, 'img_id': img_id_te, 'a1_e2': a1_e2,
             'a1_e3': a1_e3, 'a1_e4': a1_e4, 'a2_e2': a2_e2, 'a2_e3': a2_e3, 'a2_e4': a2_e4, 'sex_f_true': sex_f_true,
             'delta_suvr_comp': delta_s_comp, 't0_suvr_comp': t0_suvr_comp, 'suvrs_comp': d_suvrs_comp,
             'delta_suvr_csf': delta_s_csf, 't0_suvr_csf': t0_suvr_csf, 'suvrs_csf': d_suvrs_csf, 'exam_date': exam_dates_te}

# skip this step (include t0 data)
# no_t0_data = {}
# for k, v in meta_data.items():
#     new = v[delta_t>0]
#     no_t0_data[k] = new
no_t0_data = meta_data

sub_tt = no_t0_data['sub_id']
shuff_perm = np.random.permutation(len(sub_tt))
sub_tt = sub_tt[shuff_perm]

train_size = int(np.ceil(len(sub_tt)*0.8))
test_size = int(np.floor(len(sub_tt)*0.2))

train = np.zeros(len(sub_tt))
test = np.zeros(len(sub_tt))
for sub in np.unique(sub_tt):
    if train_size > train.sum():
        train[sub==sub_tt] = 1
    else:
        test[sub==sub_tt] = 1

train = train.astype(np.bool)
test = test.astype(np.bool)

for k in no_t0_data.keys():
    no_t0_data[k] = no_t0_data[k][shuff_perm]
no_t0_data['train'] = train
no_t0_data['test'] = test

out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'

with open(out_path, 'wb') as f:
    pickle.dump(no_t0_data, f)




print('done')
