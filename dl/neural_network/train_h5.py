if __name__ == '__main__':
    import inspect
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    parent_dir = os.path.dirname(parent_dir)
    print(parent_dir)
    sys.path.insert(0, parent_dir)

from dl.data.get_dataset import get_dataset
from dl.neural_network.resnet import ResNet50, ResNet50Reg
# from dl.neural_network.resnext import ResNext101, ResNext101Reg
from dl.neural_network.resnet_152 import ResNet152, ResNet152Reg
from dl.neural_network.train_test import train
from dl.neural_network.train_test_regression import train_reg
from dl.data.bin_equal import bin_equal
from datetime import datetime
from dl.data.project_logging import Logger, CsvWriter
import numpy as np
import torch
from torch import nn
from torch import optim
import sys
import os
import GPUtil
import pickle
from dl.neural_network.net_features_extraction import get_net_features
# set deterministic mode for cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_h5(h5_path, num_epochs=30, label_names=None, extra_info='', lr=0.01, decrease_after=10,
             rate_of_decrease=0.1, gpu_device=-1, save_pred_labels=True, test_split=0.2, pretrained=True,
             batch_size=32, binning=-1, regression=False, include_subject_ids=True, seed=-1, freeze_epochs=-1,
             use_resnext=False, use_resnet152=False, save_model=True, train_by_id = '', extract_features=False,
             threshold=1.11):
    windows_db = False
    if windows_db:
        h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data_prediction.h5'

    # chose random when -1, otherwise the selected id
    if gpu_device < 0:
        device_ids = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[],
                                         excludeUUID=[])
        gpu_device = device_ids[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    out_path = os.path.dirname(h5_path)
    time_stamp = datetime.now().strftime('%m-%d_%H-%M')
    # if there are two labels to be fetched, we assume that the first one is for training and the second one is for
    # testing
    if len(label_names) > 1:
        if include_subject_ids and not extract_features:
            data, labels, labels2, s_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True)
        elif include_subject_ids and extract_features:
            data, labels, labels2, s_ids, train_info, img_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True,
                                                          include_train_info=True)
        else:
            data, labels, labels2 = get_dataset(h5_path, label_names=label_names)
    else:
        if include_subject_ids and not extract_features:
            data, labels, s_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True)
        elif include_subject_ids and extract_features:
            data, labels, s_ids, train_info, img_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True,
                                                          include_train_info=True)
        else:
            data, labels = get_dataset(h5_path, label_names=label_names)
        # create dummy labels2. not perfect, I guess, but good enough :)
        # turns out that I actually don't need the amyloid status label, as the suvr value is obviously sufficient to
        # infer amyloid status. Still keeping it. Why, you ask? Because I can.
        labels2 = np.ones(len(labels))
    # if there is a seed, we now set a universal seed. We later on set more seeds to ensure the same shuffling/random
    # neural network initialization for all runs. We do this by multiplying the seed with prime numbers larger than
    # 100. This ensures that all seeds from 1 to 100 will not run into the same multiples.
    if not seed == -1:
        np.random.seed(seed*101)
        torch.manual_seed(seed*101)
    shuff_idxs = np.random.permutation(len(data))
    data = data[shuff_idxs]
    labels = labels[shuff_idxs]
    labels2 = labels2[shuff_idxs]
    if include_subject_ids:
        s_ids = s_ids[shuff_idxs]
    if extract_features:
        train_info = train_info[shuff_idxs]
        img_ids = img_ids[shuff_idxs]
    standard_info = f"_lr_{str(lr).replace('.', '_')}_{'pretrained' if pretrained else 'non_pretrained'}_{'reg_' if regression else ''}{str(binning)+'bins' if binning>0 else ''}{num_epochs}epochs_rod_{str(rate_of_decrease).replace('.', '_')}_da_{decrease_after}"
    if binning > 0:
        labels_backup = np.copy(labels)
        labels, break_offs = bin_equal(labels, num_bins=binning)
        with open(os.path.join(out_path, f"original_labels_and_break_offs_{time_stamp}{extra_info}{standard_info}.p"), 'wb') as f:
            pickle.dump({'original_labels': labels_backup, 'break_offs': break_offs}, f)
    # normalize data
    if len(data.shape) == 4:
        # channel wise
        data -= data.mean(axis=(0, 1, 2))
        data -= data.std(axis=(0, 1, 2))
    print(f'data mean is {data.mean()}')
    data -= data.mean()
    print(f'data std is {data.std()}')
    data /= data.std()
    # pretrained data needs input to be in the range [0,1]
    # we test, whether this is best for transfer learning as well.. -> yes, standard way is best
    if pretrained:
        # import pdb; pdb.set_trace()
        data -= data.min()
        data /= data.max()
        print(f'skewed data into min: {data.min()} and max: {data.max()}')
    # import time; time.sleep(20)
    if windows_db:
        win_db_limit = 5000
        data = data[:win_db_limit]
        labels = labels[:win_db_limit]
        labels2 = labels2[:win_db_limit]
        if include_subject_ids:
            s_ids = s_ids[:win_db_limit]
    # We try to not have the same subject in train and test set. To do so, we iteratively assign from subjects with a
    # high number of scans (max is 5) to subjects with a low number of scans (1) to train and test set. If the
    # train-test split is 4 to 1, we assign 4 subjects to train and one to test in each iteration etc. etc.
    if include_subject_ids and not extract_features:
        ratio = int(np.round((1-test_split)/test_split))
        n, c = np.unique(s_ids, return_counts=True)
        # randomness is preserved within chunks of equal count. As numpy sorts the unique ids for a reason that I
        # cannot understand, we shuffle things again:
        if seed != -1:
            np.random.seed(seed*103)
        subj_shuff_idxs = np.random.permutation(len(n))
        n = n[subj_shuff_idxs]
        c = c[subj_shuff_idxs]
        sort_unique = np.argsort(c)[::-1]
        n = list(n[sort_unique])
        c = list(c[sort_unique])
        test_size = int(len(data)*test_split)
        train_size = len(data)-test_size
        test_idxs = np.zeros(len(data)).astype(np.int)
        train_idxs = np.zeros(len(data)).astype(np.int)
        while True:
            try:
                for _ in range(int(np.floor(ratio/2))):
                    if c[0]+np.sum(train_idxs) <= train_size:
                        cc = c.pop(0)
                        nnn = n.pop(0)
                        train_idxs[s_ids == nnn] = 1
                if c[0] + np.sum(test_idxs) <= test_size:
                    cc = c.pop(0)
                    nnn = n.pop(0)
                    test_idxs[s_ids == nnn] = 1
                for _ in range(int(np.ceil(ratio/2))):
                    if c[0]+np.sum(train_idxs) <= train_size:
                        cc = c.pop(0)
                        nnn = n.pop(0)
                        train_idxs[s_ids == nnn] = 1
                if np.sum(test_idxs) == test_size and np.sum(train_idxs) == train_size:
                    break
            except:
                break
        print('data split!')
        print(f"Test set, goal of {test_size}, got {np.sum(test_idxs)}")
        print(f"Trai set, goal of {train_size}, got {np.sum(train_idxs)}")
        train_idxs = np.where(train_idxs)[0]
        test_idxs = np.where(test_idxs)[0]
        test_data = data[test_idxs]
        test_labels = labels[test_idxs]
        test_labels2 = labels2[test_idxs]
        train_data = data[train_idxs]
        train_labels = labels[train_idxs]
        train_labels2 = labels2[train_idxs]
        subj_out_path = os.path.join(out_path, f'subject_split_{time_stamp}{extra_info}{standard_info}.p')
        subj_meta_data = {'subj_id': s_ids, 'test_idxs': test_idxs, 'train_idxs': train_idxs,
                          'shuffle_permutation': shuff_idxs, 'subj_shuffle_idxs': subj_shuff_idxs}
        with open(subj_out_path, 'wb') as f:
            pickle.dump(subj_meta_data, f)
    elif extract_features:
        train_idxs = np.where(train_info)[0]
        test_idxs = np.where(train_info==False)[0]
        test_data = data[test_idxs]
        test_labels = labels[test_idxs]
        test_labels2 = labels2[test_idxs]
        train_data = data[train_idxs]
        train_labels = labels[train_idxs]
        train_labels2 = labels2[train_idxs]
    else:
        cutoff = int(len(data) * test_split)
        test_data = data[:cutoff]
        test_labels = labels[:cutoff]
        test_labels2 = labels2[:cutoff]
        train_data = data[cutoff:]
        train_labels = labels[cutoff:]
        train_labels2 = labels2[cutoff:]

    train_data = torch.from_numpy(train_data).type(torch.float32)
    test_data = torch.from_numpy(test_data).type(torch.float32)
    if len(train_data.shape) > 3:
        # we then have an extra dimension with channels
        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)
        # find out the number of channels.
        num_chan_input = train_data.shape[1]
    else:
        # 3 is the default, which we also use, when there's only one input channel
        num_chan_input = 3
    num_classes = len(np.unique(labels))

    # set separate seed for initialization
    if seed != -1:
        torch.manual_seed(seed*107)
    if regression:
        test_labels = torch.from_numpy(test_labels).type(torch.float32)
        train_labels = torch.from_numpy(train_labels).type(torch.float32)
        if use_resnext:
            Net = ResNext101Reg(pretrained=pretrained, num_classes=1, num_input=num_chan_input)
            # import pdb; pdb.set_trace()
        elif use_resnet152:
            Net = ResNet152Reg(pretrained=pretrained, num_classes=1, num_input=num_chan_input)
        else:
            Net = ResNet50Reg(pretrained=pretrained, num_classes=1, num_input=num_chan_input)
        criterion = nn.MSELoss()
        train_func = train_reg
    else:
        test_labels = torch.from_numpy(test_labels).type(torch.long)
        train_labels = torch.from_numpy(train_labels).type(torch.long)
        if use_resnext:
            Net = ResNext101(pretrained=pretrained, num_classes=num_classes, num_input=num_chan_input)
        elif use_resnet152:
            Net = ResNet152(pretrained=pretrained, num_classes=num_classes, num_input=num_chan_input)
        else:
            Net = ResNet50(pretrained=pretrained, num_classes=num_classes, num_input=num_chan_input)
        criterion = nn.NLLLoss()
        train_func = train

    Net.cuda()
    log_path = os.path.join(out_path, f"training_log_{time_stamp}{extra_info}{standard_info}.txt")
    sys.stdout = Logger(log_path)
    csv_path = os.path.join(out_path, f"train_test_accuracy_{time_stamp}{extra_info}{standard_info}.csv")
    header = ['test_acc', 'train_acc', 'train_loss', 'epoch']
    if len(label_names) > 1:
        header.extend(['test_label_acc_train', 'test_label_acc_test'])
    csv_writer = CsvWriter(csv_path, header=header)
    # set separate seed to ensure the same batch generation, as seed for neural network initialization might now be
    # varying depending on neural network architecture
    if seed != -1:
        np.random.seed(seed*109)
        torch.manual_seed(seed*109)
    for i in range(num_epochs):
        # we freeze all parameters for some epochs to fine tune the last layer first
        if freeze_epochs > 0:
            if i in range(freeze_epochs):
                Net.freeze_except_fc()
            elif i == freeze_epochs:
                Net.unfreeze_all()
        if i % decrease_after == 0:
            if i > 0:
                lr = lr*rate_of_decrease
            print(f"Trainig for {decrease_after} epochs with a learning rate of {lr}..")
        optimizer = optim.Adam(Net.parameters(), lr=lr)
        train_result = train_func(batch_size=batch_size, train_data=train_data, train_labels=train_labels, test_data=test_data,
                                  test_labels=test_labels, Net=Net, optimizer=optimizer, criterion=criterion,
                                  test_interval=1, epochs=1, dim_in='default')
        Net, test_acc, test_pred_label, train_acc, train_loss, train_pred_label = train_result
        if len(label_names) > 1 and not regression:
            test_label_acc_test = np.mean((test_pred_label[:, 0] >= num_classes/2) == (test_pred_label[:, 1] >= threshold))
            test_label_acc_train = np.mean((train_pred_label[:, 0] >= num_classes / 2) == (train_pred_label[:, 1] >= num_classes / 2))
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i,
                                 test_label_acc_train=test_label_acc_train, test_label_acc_test=test_label_acc_test)
            print(f"Amyloid status accuracy is {test_label_acc_test * 100:.2f} percent for test and {test_label_acc_train * 100:.2f} percent for train")
        elif len(label_names) > 1 and regression:
            test_label_acc_test = np.mean((test_pred_label[:, 0] >= threshold) == (test_pred_label[:, 1] >= threshold))
            test_label_acc_train = np.mean((train_pred_label[:, 0] >= threshold) == (train_pred_label[:, 1] >= threshold))
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i,
                                 test_label_acc_train=test_label_acc_train, test_label_acc_test=test_label_acc_test)
            print(f"Amyloid status accuracy is {test_label_acc_test * 100:.2f} percent for test and {test_label_acc_train * 100:.2f} percent for train")
        else:
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i)
        if save_pred_labels:
            pickle_fn = os.path.join(out_path, f"epoch_{i}_pred_labels_train_test_epoch_{time_stamp}{extra_info}{standard_info}.p")
            pickle_object = {'train': train_pred_label, 'test': test_pred_label}
            with open(pickle_fn, 'wb') as f:
                pickle.dump(pickle_object, f)
        print(f"Test accuracy is {test_acc * 100:.2f} percent")
    if extract_features:
        features_dict = get_net_features(Net, data, labels, img_ids)
        with open(os.path.join(out_path, 'fc_activations.pickle'), 'wb') as f:
            pickle.dump(features_dict, f)
    if save_model:
        model_out_path = os.path.join(out_path, f'resnet_model_{time_stamp}{extra_info}{standard_info}.pth')
        torch.save(Net, model_out_path)


if __name__ == '__main__':
    # this is done to run things from console
    windows_db = False
    if not windows_db:
        seed = 10
        job = {'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use': True, 'batch_size': 4}
        job = {'extra_info': '', 'pretrained': True, 'label_names': ['label_amyloid'], 'lr': 0.0001, 'seed': seed, 'save_model': False, 'use_resnet152': True, 'batch_size': 32}
        fpath = '/scratch/reith/fl/experiments/seeds_resnet152/seed_10/dist_10/slice_data_subj.h5'
        train_h5(fpath, **job)
    else:
        train_h5(r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\incl_subjects_site_one_slices_dataset\slice_data_subj.h5',
                 pretrained=False, extra_info='', lr=0.001, regression=True, label_names=['label_suvr', 'label_amyloid'],
                 num_epochs=1, seed=1, extract_features=True, batch_size=8)
    # label_names=['label_suvr', 'label_amyloid'],




