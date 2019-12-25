import numpy as np
import h5py
import os
import scipy.misc


def get_dataset(h5_path, label_names=['label_amyloid'], limit=-1, include_subjects=False, include_train_info=False):
    data = h5py.File(h5_path, 'r')
    arrs = []
    labels = []
    if include_subjects:
        label_names.append('rid')
    if include_train_info:
        label_names.append('train_data')
        label_names.append('img_id')
    for i in range(len(label_names)):
        labels.append([])
    for k in data.keys():
        arrs.append(data[k][()])
        for i, l in enumerate(label_names):
            labels[i].append(data[k].attrs[l])
    labels = [np.array(l) for l in labels]
    arrs = np.array(arrs)
    return (arrs, *labels)


if __name__ == '__main__':
    fp = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5'
    # dset, labels = get_dataset(fp)
    dset, l1, l2 = get_dataset(fp, label_names=['label_amyloid', 'label_suvr'])
    out_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl'
    out_path = os.path.join(out_path, 'data_image')
    os.makedirs(out_path, exist_ok=True)
    for arr, suvr in zip(dset, l2):
        fname = f"slice_50_suvr_{str(suvr).replace('.', '_')}.png"
        fpath = os.path.join(out_path, fname)
        scipy.misc.imsave(fpath, arr)
    print('done')
