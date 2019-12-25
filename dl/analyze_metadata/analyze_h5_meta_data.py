import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
import h5py
import plotly.express as px


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
in_path2 = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'

with open(in_path, 'rb') as f:
    data_pickle = pickle.load(f)

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


suvr = label_suvr[~train_data]
time = scan_time[~train_data]
sub = sub_id[~train_data]

delta_s = []
delta_t = []
d_suvrs = []
for s in np.unique(sub):
    su = suvr[sub==s]
    ti = time[sub==s]
    if ti.min() >0:
        if len(ti) > 1:
            ti -= ti.min()
        else:
            continue
    delta_su = su-su[ti.argmin()]
    if delta_su.min() < -0.2:
        print('db')
    delta_s.extend(delta_su)
    delta_t.extend(ti)
    d_suvrs.extend(su)

d_suvr_comp = d_suvrs
delta_comp = delta_s

suvr = label_suvr[~train_data]
time = scan_time[~train_data]
sub = sub_id[~train_data]

delta_s = []
delta_t = []
d_suvrs = []
for s in np.unique(sub):
    su = suvr[sub==s]
    ti = time[sub==s]
    if ti.min() >0:
        if len(ti) > 1:
            ti -= ti.min()
        else:
            continue
    delta_su = su-su[ti.argmin()]
    if delta_su.min() < -0.2:
        print('db')
    delta_s.extend(delta_su)
    delta_t.extend(ti)
    d_suvrs.extend(su)

delta_s = np.array(delta_s)
delta_t = np.array(delta_t)
d_suvrs = np.array(d_suvrs)
delta_comp = np.array(delta_comp)

t_sum = np.array(d_suvrs)>1.11
t_comp = np.array(d_suvr_comp)>0.79
print(np.mean(delta_comp[t_comp]))
print(np.mean(delta_comp[~t_comp]))
t_sum_names = ['positive' if x else 'negative' for x in t_sum]
t_comp_names = ['positive' if x else 'negative' for x in t_comp]


print(np.mean(delta_s[t_sum]))
print(np.mean(delta_s[~t_sum]))
df_refnorm = pd.DataFrame({'Amyloid_status': t_comp_names, 'Delta SUVR REFNORM': delta_comp, 'Delta time (years)': delta_t})
df_cerebnorm = pd.DataFrame({'Amyloid_status': t_sum_names, 'Delta SUVR CEREBNORM': delta_s, 'Delta time (years)': delta_t})

df_cerebnorm = df_cerebnorm.melt(id_vars='Delta time (years)')
fig = px.scatter(df_cerebnorm, x='Delta time (years)', y='Delta SUVR CEREBNORM', color='Amyloid_status', trendline='ols')
fig.show()
results = px.get_trendline_results(fig)
print(results)
results.query("Amyloid_status == 'negative'").px_fit_results.iloc[0].summary()
results.query("Amyloid_status == 'positive'").px_fit_results.iloc[0].summary()


fig = px.scatter(df_refnorm, x='Delta time (years)', y='Delta SUVR REFNORM', color='Amyloid_status', trendline='ols')
fig.show()
results = px.get_trendline_results(fig)
print(results)
results.query("Amyloid_status == 'negative'").px_fit_results.iloc[0].summary()
results.query("Amyloid_status == 'positive'").px_fit_results.iloc[0].summary()

print('done')

# fig = plt.figure()
# # plt.scatter(delta_t[(delta_t>0) & (d_suvrs<0.79)], delta_s[(delta_t>0) & (d_suvrs<0.79)], label='Amyloid negative', s=1, alpha=0.5)
# plt.scatter(delta_t[(delta_t>0) & (d_suvrs>0.79)], delta_s[(delta_t>0) & (d_suvrs>0.79)], label='Amyloid positive', s=1, alpha=0.5)
# plt.legend()
# plt.ylabel('Delta SUVR composite')
# plt.xlabel('Delta time (years)')
#
# out_path = r'C:\Users\Fabian\Desktop\Greg'
# fname = 'suvr_composite_scatter_plot_positive'
# fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)


# out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_exam_times.pickle'
#
# with open(out_path, 'wb') as f:
#     pickle.dump(labels, f)
