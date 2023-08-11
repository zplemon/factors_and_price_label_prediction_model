#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import pandas as pd

 
path_1 = 'D:\\cicc_reits_pred\\fac_w_hf.csv'
path_2 = 'D:\\cicc_reits_pred\\close_df.csv'

factors = pd.read_csv(path_1,index_col=[0])
close = pd.read_csv(path_2,index_col=[0])

sec_name_list = []
for name in close.columns[1:]:
    name = name[:9]
    sec_name_list = sec_name_list + [name]


def get_ln_ret(a, b):
    if np.isnan(b):
        return b
    if math.isclose(b, 0, abs_tol=1e-15) == True:
        return np.nan
    return np.log(a/b)


ln_ret_df = pd.DataFrame(close['time'])
for i in sec_name_list:
    close_temp = close.copy(deep=True)
    close_temp[i+'_shift'] = close_temp[i+'_close'].shift(1)
    ln_ret_df[i] = close_temp.apply(lambda x: get_ln_ret(x[i+'_close'],x[i+'_shift']), axis=1)

ln_ret_df_lag = ln_ret_df.copy(deep=True)
ln_ret_df_lag.iloc[:,1:] = ln_ret_df.iloc[:,1:].shift(-1)

ln_ret_df_lag_2 = ln_ret_df_lag.copy(deep=True)
ln_ret_df_lag_2.iloc[:,1:] = ln_ret_df_lag.iloc[:,1:].shift(-1)

all_data = pd.merge(factors,ln_ret_df_lag_2,on='time',how='inner')

sec_dep_data = pd.DataFrame()
sec_dep_data = all_data.loc[:,'508000.SH_high_prev':]
sec_dep_data['time'] = all_data['time']
#sec_dep_data = sec_dep_data.dropna(subset=sec_dep_data.columns[:-1],how='all')

time_list = list(sec_dep_data['time'])
sec_dep_f = ['high_prev','low_prev','close_prev','day_avg_prev','vol_prev','amt_prev','turnover_prev',
             'rskew_prev','downward_ratio_prev','reverse_prev','tail_vol_ratio_prev','pvol_corr_prev',
             'large_trans_ret_prev','ind_all_diff_prev','ind_all_dacce_prev','ind_highway_diff_prev',
             'ind_highway_dacce_prev','ind_park_diff_prev','ind_park_dacce_prev']
day_data_dict = {}

warnings.simplefilter(action= 'ignore', category=pd.errors.PerformanceWarning)
for day in time_list:
    ret_df = sec_dep_data.loc[sec_dep_data['time']==day,sec_name_list]
    ret_df = ret_df.T
    ret_df.columns = ['ln_ret']
    for col in sec_dep_data.columns[:-29]:
        name = col[:9]
        ret_df.loc[name,col[10:]] = sec_dep_data.loc[sec_dep_data['time']==day,col].values
    day_data_dict[day] = ret_df

ic_df = pd.DataFrame(index = time_list, columns = sec_dep_f)
ric_df = pd.DataFrame(index = time_list, columns = sec_dep_f)
for day in time_list:
    ret_df = day_data_dict[day]
    ic_df.loc[day] = ret_df.corr(method='pearson',numeric_only=True)['ln_ret'].iloc[1:]
    ric_df.loc[day] = ret_df.corr(method='spearman',numeric_only=True)['ln_ret'].iloc[1:]


def get_pos_skip_na(x):
    if np.isnan(x):
        return x
    return (x > 0)


IC_cl = ic_df.dropna(how='all')
IC_mean = IC_cl.mean(skipna=True)
IC_std = IC_cl.std(skipna=True)
ICIR = IC_mean/IC_std
IC_pos_df = IC_cl.applymap(get_pos_skip_na)
p_IC_pos = IC_pos_df.mean(skipna=True)

rIC_cl = ric_df.dropna(how='all')
rIC_mean = rIC_cl.mean(skipna=True)
rIC_std = rIC_cl.std(skipna=True)
rICIR = rIC_mean/rIC_std
rIC_pos_df = rIC_cl.applymap(get_pos_skip_na)
p_rIC_pos = rIC_pos_df.mean(skipna=True)

result_table = pd.DataFrame(columns = ["IC_mean","ICIR","p(IC>0)","rIC_mean","rICIR","p(rIC>0)"])
result_table["IC_mean"] = IC_mean
result_table["ICIR"] = ICIR
result_table["p(IC>0)"] = p_IC_pos
result_table["rIC_mean"] = rIC_mean
result_table["rICIR"] = rICIR
result_table["p(rIC>0)"] = p_rIC_pos

result_table.to_csv('eval_hfm_lag1.csv')