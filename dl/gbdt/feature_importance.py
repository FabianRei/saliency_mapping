from glob import glob
import os
import pickle
import numpy as np


target_folder = r'C:\Users\Fabian\stanford\gbdt\analysis'
target_file = 'all_results.pickle'
folders = glob(os.path.join(target_folder, '157*'))

folder = folders[-1]
target = os.path.join(folder, target_file)
with open(target, 'rb') as f:
    data = pickle.load(f)

gbms = data['gbm']
x_names = data['x_names']
preds = data['predictions']
labs = data['labels']
labs = np.concatenate(labs)
preds = np.concatenate(preds)
sort_idxs = np.argsort(preds)[::-1]
sort_idxs2 = np.argsort(labs)[::-1]
labs = labs[sort_idxs]
preds = preds[sort_idxs]
print(sort_idxs[:25])
print(labs[:25])
print(preds[:25])
fis = []
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