import csv
import numpy as np
import pickle
import os
import pandas as pd


def write_csv_row(resultCSV, res_dict, headers):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(res_dict)


in_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\csf_pet_diffs.pickle'

with open(in_path, 'rb') as f:
    data = pickle.load(f)

abetas = np.array(data['abetas'])
rids = np.array(data['rids'])
csf_d = np.array(data['csf_examdates'])
pet_d = np.array(data['pet_examdates'])
cereb = np.array(data['suvrs_cerebnorm'])
pet_csf = np.array(data['suvrs_csf'])
comp = np.array(data['suvrs_omposit'])


headers = ['RID', 'CSF_EXAMDATE', 'CSF_ABETA', 'DELTA_CSF_ABETA', 'PET_EXAMDATE', 'PET_SUVR_CEREBNORM', 'DELTA_PET_SUVR_CEREBNORM',
           'PET_SUVR_COMPOSIT', 'DELTA_PET_SUVR_COMPOSIT',
           'PET_SUVR_CSF', 'DELTA_PET_SUVR_CSF', 'PET_CSV_DIFF_DAYS', 'TIME_DIFF_FIRST_EXAM_PET_YEARS', 'TIME_DIFF_FIRST_EXAM_CSF_YEARS']
csv_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\CSF_PET.csv'
if os.path.exists(csv_path):
    os.remove(csv_path)

for rid in np.unique(rids):
    res = {}
    res['ab'] = abetas[rid==rids]
    res['ri'] = rids[rid==rids]
    res['d_csf'] = csf_d[rid==rids]
    res['d_pet'] = pet_d[rid==rids]
    res['ce'] = cereb[rid==rids]
    res['co'] = comp[rid==rids]
    res['cs'] = pet_csf[rid==rids]
    sort_idxs = np.argsort(res['d_pet'])
    # sort
    for k in res.keys():
        res[k] = res[k][sort_idxs]
    # get dates
    pet_dates = []
    for d in res['d_pet']:
        d = str(pd.to_datetime(d, unit='s').date())
        pet_dates.append(d)
    res['date_pet'] = np.array(pet_dates)
    csf_dates = []
    for d in res['d_csf']:
        d = str(pd.to_datetime(d, unit='s').date())
        csf_dates.append(d)
    res['date_csf'] = np.array(csf_dates)
    # delta suvrs
    res['del_ce'] = res['ce'] - res['ce'][0]
    res['del_co'] = res['co'] - res['co'][0]
    res['del_cs'] = res['cs'] - res['cs'][0]
    # delta csf
    res['del_ab'] = res['ab'] - res['ab'][0]
    # time diff csf/pet
    res['time_diff_pet_csf'] = np.abs(res['d_pet']-res['d_csf'])/24/60/60
    res['time_diff_pet'] = np.abs(res['d_pet']-res['d_pet'][0])/(24*60*60*365)
    res['time_diff_csf'] = np.abs(res['d_csf'] - res['d_csf'][0]) / (24 * 60 * 60 * 365)
    for i in range(len(res['ri'])):
        csv_dict = {'RID': res['ri'][i], 'CSF_EXAMDATE': res['date_csf'][i], 'CSF_ABETA': res['ab'][i], 'DELTA_CSF_ABETA': res['del_ab'][i],
                'PET_EXAMDATE': res['date_pet'][i], 'PET_SUVR_CEREBNORM': res['ce'][i], 'DELTA_PET_SUVR_CEREBNORM': res['del_ce'][i],
               'PET_SUVR_COMPOSIT': res['co'][i], 'DELTA_PET_SUVR_COMPOSIT': res['del_co'][i], 'PET_SUVR_CSF': res['cs'][i],
                'DELTA_PET_SUVR_CSF': res['del_cs'][i], 'PET_CSV_DIFF_DAYS': res['time_diff_pet_csf'][i],
               'TIME_DIFF_FIRST_EXAM_PET_YEARS': res['time_diff_csf'][i], 'TIME_DIFF_FIRST_EXAM_CSF_YEARS': res['time_diff_pet'][i]}
        write_csv_row(csv_path, csv_dict, headers)

    print('nice')





print('nice')