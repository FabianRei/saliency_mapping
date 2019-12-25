import pickle
import pandas as pd
import numpy as np
import re
import plotly.express as px




def get_nearest_csf(rid, examdate):
    abetas = []
    dates = []
    # mk9 idxs
    idxs = mk9['RID'] == rid
    if np.sum(idxs) > 0:
        idxs_num = np.where(idxs)[0]
        abeta_limited = list(mk9['ABETA'][idxs])
        for a, i in zip(abeta_limited, idxs_num):
            if a == '>1700':
                ab = mk9['COMMENT'][i]
                num = re.findall(r'\d{3,10}', ab)[0]
                abetas.append(float(num))
            else:
                abetas.append(float(a))
            date = mk9['EXAMDATE'][i]
            date = date.timestamp()
            dates.append(date)
    # mk10 idxs
    mk10_idxs = mk10['RID'] == rid
    if np.sum(idxs) > 0:
        abeta = mk10['ABETA42'][mk10_idxs]
        date = mk10['DRAWDATE'][mk10_idxs]
        for a, d in zip(abeta, date):
            abetas.append(float(a))
            dates.append(d.timestamp())
    if len(dates) == 0:
        # sucks to not have a date
        return -1, -1, -1
    else:
        dates = np.array(dates)
        date_diffs = dates - examdate
        date_diffs = np.abs(date_diffs)
        min_idx = date_diffs.argmin()
        if np.sum(np.isnan(abetas)) > 0:
            return -1, -1, -1
        return abetas[min_idx], date_diffs[min_idx]/(365*24*60*60), dates[min_idx]



out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'
mk9_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\UPENNBIOMK9_04_19_17.csv'
mk10_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\UPENNBIOMK10_07_29_19.csv'
mk9 = pd.read_csv(mk9_path, parse_dates=['EXAMDATE'])
mk10 = pd.read_csv(mk10_path, parse_dates=['DRAWDATE'])
with open(out_path, 'rb') as f:
    data = pickle.load(f)

ex_dates = data['exam_date']
rids = data['sub_id']
time_deltas = data['delta_time']
suvrs_csf = data['suvrs_csf']
suvrs_comp = data['suvrs_comp']
suvrs_cerebnorm = data['suvrs']

nearest_csf_times = []
nearest_csf_vals = []
nearest_csv_examdates = []
for ex, ri in zip(ex_dates, rids):
    a, e, d = get_nearest_csf(ri, ex)
    nearest_csf_times.append(e)
    nearest_csf_vals.append(a)
    nearest_csv_examdates.append(d)

nearest_csf_times = np.array(nearest_csf_times)
nearest_csf_vals = np.array(nearest_csf_vals)
nearest_csv_examdates = np.array(nearest_csv_examdates)

diff_idxs = np.where((nearest_csf_times >= 0) & (nearest_csf_times < 30/365))[0]

selected_abetas = nearest_csf_vals[diff_idxs]
selected_times = time_deltas[diff_idxs]
selected_rids = rids[diff_idxs]
selected_csf_examdates = nearest_csv_examdates[diff_idxs]
selected_pet_examdates = ex_dates[diff_idxs]
selected_csv_pet_timediffs = nearest_csf_times[diff_idxs]
selected_csf_suvrs = suvrs_csf[diff_idxs]
selected_comp_suvrs = suvrs_comp[diff_idxs]
selected_cerebnorm_suvrs = suvrs_cerebnorm[diff_idxs]

# save to pickle
# out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\csf_pet_diffs.pickle'
# with open(out_path, 'wb') as f:
#     pickle.dump({'abetas': selected_abetas, 'rids': selected_rids, 'csf_examdates': selected_csf_examdates, 'pet_examdates': selected_pet_examdates,
#                  'suvrs_cerebnorm': selected_cerebnorm_suvrs, 'suvrs_csf': selected_csf_suvrs, 'suvrs_omposit': selected_comp_suvrs}, f)

# visualization
amyloid_status = (selected_abetas>192)
viz_abetas = []
viz_times = []
viz_amyloid_status = []


for r in np.unique(selected_rids):
    times_r = selected_times[r==selected_rids]
    abetas_r = selected_abetas[r==selected_rids]
    am_stats = amyloid_status[r==selected_rids]
    if len(times_r) <= 1:
        continue
    results = abetas_r - abetas_r[times_r.argmin()]
    viz_abetas.extend(results)
    viz_times.extend(times_r)
    viz_amyloid_status.extend(am_stats)


viz_abetas = np.array(viz_abetas); viz_times = np.array(viz_times); viz_amyloid_status = np.array(viz_amyloid_status).astype(np.bool)

amyloid_status_names = np.array(['positive' if x else 'negative' for x in viz_amyloid_status])
df_csf = pd.DataFrame({'Amyloid status': amyloid_status_names[viz_times>0], 'Delta CSF': viz_abetas[viz_times>0], 'Delta time (years)': viz_times[viz_times>0]})

# df_csf = df_csf.melt(id_vars='Delta time (years)')
# fig = px.scatter(df_cerebnorm, x='Delta time (years)', y='Delta SUVR CEREBNORM', color='Amyloid_status', trendline='ols')
fig = px.scatter(df_csf, x='Delta time (years)', y='Delta CSF', color='Amyloid status', trendline='ols')
fig.show()