import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter

def get_key(key_ids, id):
    for k in key_ids:
        if id == k.split('_')[-1]:
            return k


def get_id(ids, key):
    for i in ids:
        if i == key.spit('_')[-1]:
            return i


# label_path = '/share/wandell/data/reith/federated_learning/labels_detailled.pickle'
label_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal.pickle'
with open(label_path, 'rb') as f:
    labels = pickle.load(f)
# excel_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\Scanner information.gz.xls'
# with open(excel_path, 'rb') as f:
#     scan_sheet = pd.read_excel(f)

scanners = []
amyloid = []
rcf = []
img_ids = []
names = []
slices = []
site = []
suvr = []
subject_ids = []
examdate_posix = []
examdate = []
ages = []
key_ids = []
# scan_keys = np.array(scan_sheet['Scanner'])
# scanner_keys = np.array([','.join(f.split(',')[:-1]) for f in scan_keys])
# scan_numbers = np.array(scan_sheet['Type'])
ap1 = []
ap2 = []

for k, v in labels.items():
    key_ids.append(k)
    scanners.append(f"{v['manufacturer']}, {v['model']}")
    amyloid.append(v['label'])
    rcf.append([v['rows'], v['columns'], v['frames']])
    img_ids.append(v['img_id'])
    names.append(k)
    slices.append(v['slices'])
    site.append(v['site'])
    suvr.append(v['label_suvr'])
    subject_ids.append(v['rid'])
    examdate_posix.append(v['examdate_posix'])
    examdate.append(v['examdate'])
    ages.append(float(v['age']))
    ap1.append(v['apoea1'])
    ap2.append(v['apoea2'])

ap1 = np.array(ap1)
ap2 = np.array(ap2)
key_ids = np.array(key_ids)
site = np.array(site)
names = np.array(names)
slices = np.array(slices)
img_ids = np.array(img_ids)
scanners = np.array(scanners)
amyloid = np.array(amyloid)
rcf = np.array(rcf)
subject_ids = np.array(subject_ids)
suvr = np.array(suvr)
examdate_posix = np.array(examdate_posix)
types = []
examdate = np.array(examdate)
ages = np.array(ages)

n, c = np.unique(subject_ids, return_counts=True)
subs = n[c>1]

not_train_ids = []
sub_dict = {}
sub_dict_ids = {}
id_dict = {}
subs = np.random.permutation(subs)
count = 0
not_train_sub_ids = []

# csv_filepath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\times_complete.csv'

# writer = CsvWriter(csv_filepath, header=['subject_id', 'time', 'suvr'], delim=',')
for ii, sub in enumerate(subs):
    if sub==31:
        print('db')
    times = examdate_posix[subject_ids == sub]
    time_ids = img_ids[subject_ids == sub]
    suvrs = suvr[subject_ids == sub]>1.11
    times -= times.min()
    times /= (365 * 24 * 60 * 60)
    if ii > 600 and not (len(np.unique(suvrs)) == 2):
        continue
    if sub==4169:
        print('db')
    if len(np.unique(suvrs)) == 2 or True:
        count += 1
        suvr_times = suvr[subject_ids == sub][np.argsort(times)]
        # times = times[np.argsort(times)]
        # for su, ti in zip(suvr_times, times):
        #     writer.write_row(subject_id=sub, time=ti, suvr=su)
        print(times[np.argsort(times)], suvr[subject_ids == sub][np.argsort(times)], count)
    for i, j in zip(time_ids, times):
        id_dict[i] = j
    sub_dict[sub] = times
    sub_dict_ids[sub] = time_ids
    not_train_ids.extend(img_ids[subject_ids == sub])
    not_train_sub_ids.append(sub)

train_ids = [i for i in img_ids if i not in not_train_ids]
train_sub_ids = [i for i in subject_ids if i not in not_train_sub_ids]

# labels['id_dict_times'] = id_dict
# labels['sub_dict_times'] = sub_dict
# labels['train_ids'] = train_ids
# labels['train_sub_ids'] = train_sub_ids

for k in key_ids:
    labels[k]['train_data'] = False
    labels[k]['scan_time'] = -1

for id in train_ids:
    k = get_key(key_ids, id)
    labels[k]['train_data'] = True


for id in not_train_ids:
    k = get_key(key_ids, id)
    labels[k]['scan_time'] = id_dict[id]



out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal_times_fixed.pickle'

with open(out_path, 'wb') as f:
    pickle.dump(labels, f)
