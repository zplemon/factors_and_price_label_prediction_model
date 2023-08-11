#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.stats as st
import statsmodels.api as sm
import statistics
from itertools import chain
import pandas as pd
import os

pairs_required = 10 #横截面所需对数

# ### Import data并初步处理数据
path = 'D:\\cicc-因子\\因子研究\\2023_April\\'
files = os.listdir(path)
 
data = pd.DataFrame()
for file in files:
    if file.endswith('.csv'):
        data = pd.concat([data, pd.read_csv(path + file)])


def get_time_str(x,start,end):
    return x[start:end]


data['day'] = data['time_stamp'].apply(lambda x: get_time_str(x,1,11))
day_list = data['day'].unique()


# data是原始数据，一切修改只跑以下code

# take all securities starting with "180" and "508"
def ID_filter(x):
    if x.startswith(' "508') or x.startswith(' "180'):
        return True
    else:
        return False


data_new = data[data["instrumentID"].apply(lambda x: ID_filter(x))]
ID_list = data_new["instrumentID"].unique()


# create a dictionary where key is instrumentID
sec_dict = {}
for i in ID_list:
    sec_dict[i] = data_new[data_new["instrumentID"] == i]

pd.set_option('mode.chained_assignment', None)
for i in ID_list:
    #time_stamp_cut的格式：YYYY-MM-DD HH:MM:SS.sss
    sec_dict[i]['time_stamp_cut'] = sec_dict[i]['time_stamp'].apply(lambda x: get_time_str(x,1,-1))


# 发现时间点交集为空
time_unique = sec_dict[ID_list[0]]['time_stamp_cut']
for i in ID_list:
    time_unique = list(set(time_unique).intersection(sec_dict[i]['time_stamp_cut']))
time_unique


# 寻找最大交易时间
time_unique = sec_dict[ID_list[0]]['time_stamp'].unique()
for i in ID_list:
    time_unique = list(set(time_unique).union(sec_dict[i]['time_stamp_cut']))
min(time_unique)
#最大交易时间为17:00:33.719


# 构建数据结构-eg: day_dict[day_list[0]][ID_list[0]]['AvgLastPrice']
day_dict={}


def day_filter(x,day):
    if x.startswith(day):
        return True
    else:
        return False


for j in day_list:
    sec_dict_d1 = {}
    for i in ID_list:
        sec_dict_d1[i] = sec_dict[i][sec_dict[i]['time_stamp_cut'].apply(lambda x: day_filter(x,j))]
    day_dict[j] = sec_dict_d1


# filter out今天交易数据太少的securities
security_name_dict = {}
for j in day_list:
    data_num = {}
    for i in ID_list:
        data_num[i] = day_dict[j][i].shape[0]
    data_num_mean = statistics.mean(data_num.values())
    data_num_std = statistics.stdev(data_num.values())
    security_num_c = [i for i in ID_list if data_num[i] > data_num_mean-data_num_std]
    security_name_dict[j] = security_num_c
# got X securities after filtering out trade_less ones


day_dict_cl = {}
for j in day_list:
    day_dict_cl[j] = {key: day_dict[j][key] for key in security_name_dict[j]}


# 先把每个security每个时点的factor和return算出来，再以3秒为一个切片，统计该切片内所有stock的factors，return并排序

# ### 计算所有security available dates的factor和return

# ####  最新成交价距离基准价距离: bl_dist
# ####  最新成交价距中间价距离: ml_dist
# ####  切片内成交均价距基准价距离: ba_dist
# ####  市场深度：depth
# ####  成交量占上一行情切片堆积比率：tVol_depth
# ####  成交量占单边堆积比率: tVol_abVol


from datetime import datetime
import math


def trans_to_datetime(x):
    return datetime.strptime(x+'000', '%Y-%m-%d %H:%M:%S.%f')


# create a new dict to store calculated factors and returns for each security
sec_dict_w_f = {}


def get_seconds(x):
    return x.total_seconds()


def get_bl_dist(bp,lp):
    if lp==0:
        return np.nan
    else:
        return (bp-lp)/lp


def get_ml_dist(mp,lp):
    if lp==0:
        return np.nan
    else:
        return (mp-lp)/lp


def avgLastPrice_ratio(avgLastPrice, midPrice, CurrentTradeVol, depth_shift, askVol1_shift, askVol2_shift,askVol3_shift, askVol4_shift,askVol5_shift, 
                       bidVol1_shift,bidVol2_shift,bidVol3_shift,bidVol4_shift,bidVol5_shift):
    if avgLastPrice == midPrice:
        if depth_shift == 0:
            return CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/depth_shift
    elif avgLastPrice > midPrice:
        if askVol1_shift+askVol2_shift+askVol3_shift+askVol4_shift+askVol5_shift == 0:
            return  CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/ (askVol1_shift+askVol2_shift+askVol3_shift+askVol4_shift+askVol5_shift)
    else:
        if (bidVol1_shift+bidVol2_shift+bidVol3_shift+bidVol4_shift+bidVol5_shift) == 0:
            return CurrentTradeVol / 0.0000001
        else:
            return CurrentTradeVol/ (bidVol1_shift+bidVol2_shift+bidVol3_shift+bidVol4_shift+bidVol5_shift)


def get_rskew(rol_sum_r3, rol_sum_r2):
    if math.isclose(rol_sum_r2, 0, abs_tol=1e-15) == True:
        return 0
    return (60 ** (1 / 2)) * rol_sum_r3 / (rol_sum_r2 ** (3 / 2))


def get_downward_ratio(rol_sum_r2_r_neg, rol_sum_r2):
    if math.isclose(rol_sum_r2, 0, abs_tol=1e-15) == True:
        return 0
    return rol_sum_r2_r_neg / rol_sum_r2


def pvol_corr_adj(a):
    if np.isnan(a):
        return a
    elif (a > 1) or (a < -1):
        return np.nan
    else:
        return a


def get_sigmoid(a):
    if np.isnan(a):
        return a
    elif a == float("inf"):
        return a
    else:
        return 1/(1+math.exp(a))


def get_sigmoid_large(a):
    if np.isnan(a):
        return a
    elif a == float("inf"):
        return a
    else:
        return 1/(1+1+(a**1)/1+(a**2)/2+(a**3)/6+(a**4)/24+(a**5)/120)


pd.set_option('mode.chained_assignment', None)
for j in day_list:
    sec_dict_w_f[j] = {}
    for i in security_name_dict[j]:
        data_temp = day_dict_cl[j][i]
        data_temp['trade_time'] = data_temp['time_stamp_cut'].apply(lambda x: trans_to_datetime(x))
        data_temp = data_temp.sort_values(by=["trade_time"], ascending=True, ignore_index=True)
        data_temp = data_temp.dropna(axis=0, how='all')
        #关于除以的秒数的说明：上一次update时还没有交易，而这一次update时有了交易，则return的时段是update间的时段
        data_temp['seconds_to_last'] = data_temp['trade_time'].diff().apply(lambda x: get_seconds(x))
        #get midprice
        data_temp['midPrice'] = (data_temp['bidPrice1'] + data_temp['askPrice1'])/2
        #get CurrentTradeAmt
        data_temp['CurrentTradeAmt'] = data_temp['tradeAmnt'].diff()
        #get CurrentTradeVol
        data_temp['CurrentTradeVol'] = data_temp['volume'].diff()
        data_temp = data_temp[data_temp['CurrentTradeVol']!=0]
        #get AvgLastPrice
        data_temp['AvgLastPrice'] = data_temp['CurrentTradeAmt']/data_temp['CurrentTradeVol']

        #normalize benchPrice, lastPrice,
        ##get Factors
        #1. BenchPrice-already had('benchPrice'); 2. MidPrice-calculated('midPrice')
        data_temp['benchPrice_sigm'] = data_temp['benchPrice'].apply(get_sigmoid)
        data_temp['midPrice_sigm'] = data_temp['midPrice'].apply(get_sigmoid)
        #3.1 最新成交价距离基准价距离
        data_temp["bl_dist"] = data_temp.apply(lambda x: get_bl_dist(x["benchPrice"], x["lastPrice"]), axis=1)
        data_temp['bl_dist_sigm'] = data_temp['bl_dist'].apply(get_sigmoid)
        #3.2 AvgLastPrice距离基准价距离
        data_temp["b_avl_dist"] = (data_temp["benchPrice"]-data_temp["AvgLastPrice"]) / data_temp["AvgLastPrice"]
        data_temp['b_avl_dist_sigm'] = data_temp['b_avl_dist'].apply(get_sigmoid)
        #4.1 最新成交价距中间价距离
        data_temp["ml_dist"] = data_temp.apply(lambda x: get_ml_dist(x["midPrice"], x["lastPrice"]), axis=1)
        data_temp['ml_dist_sigm'] = data_temp['ml_dist'].apply(get_sigmoid)
        #4.2 AvgLastPrice距中间价距离
        data_temp["m_avl_dist"] = (data_temp["midPrice"]-data_temp["AvgLastPrice"]) / data_temp["AvgLastPrice"]
        data_temp['m_avl_dist_sigm'] = data_temp['m_avl_dist'].apply(get_sigmoid)
        #5 切片内成交均价距基准价距离
        data_temp["ba_dist"] = (data_temp["benchPrice"]-data_temp['AvgLastPrice']) / data_temp["benchPrice"]
        data_temp['ba_dist_sigm'] = data_temp['ba_dist'].apply(get_sigmoid)
        #6. 市场深度
        data_temp["depth"] = (data_temp["askVolume1"]+data_temp["askVolume2"]+data_temp["askVolume3"]+data_temp["askVolume4"]+data_temp["askVolume5"]+
                              data_temp["bidVolume1"]+data_temp["bidVolume2"]+data_temp["bidVolume3"]+data_temp["bidVolume4"]+data_temp["bidVolume5"]) /2
        data_temp['depth_sigm'] = data_temp['depth'].apply(get_sigmoid_large)
        cols = ["askVolume1", "askVolume2","askVolume3","askVolume4","askVolume5",
                "bidVolume1", "bidVolume2","bidVolume3","bidVolume4","bidVolume5", "depth"]
        for col in cols:
            data_temp[col + "_shift"] = data_temp[col].shift()
        #7. 成交量占上一行情切片堆积比率
        data_temp["tVol_depth"] = data_temp["CurrentTradeVol"]/ [m if m != 0 else 0.0000001 for m in data_temp["depth_shift"]]
        data_temp['tVol_depth_sigm'] = data_temp['tVol_depth'].apply(get_sigmoid_large)
        #8. 成交量占单边堆积比率
        data_temp["tVol_abVol"] = data_temp.apply(lambda x: avgLastPrice_ratio(x['AvgLastPrice'], x['midPrice'],x["CurrentTradeVol"], x["depth_shift"] ,x["askVolume1_shift"],x["askVolume2_shift"],x["askVolume3_shift"],x["askVolume4_shift"],x["askVolume5_shift"],
                                                                             x["bidVolume1_shift"], x["bidVolume2_shift"], x["bidVolume3_shift"],  x["bidVolume4_shift"],  x["bidVolume5_shift"]),  axis=1)
        data_temp['tVol_abVol_sigm'] = data_temp['tVol_abVol'].apply(get_sigmoid_large)
        starttime = datetime.strptime(j+' 09:30:00.000000', '%Y-%m-%d %H:%M:%S.%f')
        #get Return, 除以秒数
        data_temp["Return"] = data_temp["AvgLastPrice"].pct_change()/data_temp["seconds_to_last"]
        data_temp['Return_sigm'] = data_temp['Return'].apply(get_sigmoid)
        data_temp["Return_p_1"] = data_temp["Return"].shift(-1)
        data_temp['Return_p_1_sigm'] = data_temp['Return_p_1'].apply(get_sigmoid)

        win_len_f = dt.timedelta(milliseconds=3 * 20 * 1000)
        data_temp["Return3"] = data_temp["Return"] ** 3
        data_temp["Return2"] = data_temp["Return"] ** 2
        data_temp['ret_neg'] = data_temp["Return"] < 0
        data_temp['Return2_r_neg'] = data_temp['ret_neg'] * data_temp["Return2"]
        data_temp['AvgLastPrice_shift'] = data_temp['AvgLastPrice'].shift()
        data_temp['AvgLastPrice_shift_ratio'] = data_temp['AvgLastPrice'] / data_temp['AvgLastPrice_shift']

        data_temp3 = data_temp
        data_temp3['index_org'] = data_temp3.index
        data_temp3 = data_temp3.set_index('trade_time', drop=False)

        data_temp3['rol_sum_r3'] = data_temp3["Return3"].rolling('60s').sum()
        data_temp3['rol_sum_r2'] = data_temp3["Return2"].rolling('60s').sum()
        data_temp3['rol_sum_r2_r_neg'] = data_temp3["Return2_r_neg"].rolling('60s').sum()
        data_temp3['ALP_ratio_prod'] = data_temp3['AvgLastPrice_shift_ratio'].rolling('60s').apply(np.nanprod, raw=True)

        # 9. 高频偏度
        data_temp3["rskew"] = data_temp3.apply(lambda x: get_rskew(x['rol_sum_r3'], x['rol_sum_r2']), axis=1)
        data_temp3['rskew'] = np.real(data_temp3['rskew'])
        data_temp3['rskew_sigm'] = data_temp3['rskew'].apply(get_sigmoid)
        # 10. 下行波动占比
        data_temp3["downward_ratio"] = data_temp3.apply(
            lambda x: get_downward_ratio(x['rol_sum_r2_r_neg'], x['rol_sum_r2']), axis=1)
        data_temp3['downward_ratio_sigm'] = data_temp3['downward_ratio'].apply(get_sigmoid)
        # 11. 改进反转
        data_temp3["reverse"] = data_temp3['ALP_ratio_prod'] - 1
        data_temp3['reverse_sigm'] = data_temp3['reverse'].apply(get_sigmoid)
        # 12. 量价相关性
        data_temp3["pvol_corr"] = data_temp3['AvgLastPrice'].rolling('60s').corr(data_temp3['CurrentTradeVol'],numeric_only=True)
        data_temp3["pvol_corr"] = data_temp3["pvol_corr"].apply(pvol_corr_adj)
        data_temp3['pvol_corr_sigm'] = data_temp3['pvol_corr'].apply(get_sigmoid)
        data_temp3 = data_temp3.set_index('index_org')

        sec_dict_w_f[j][i] = data_temp3


sec_name_list_dup = list(chain.from_iterable(list(security_name_dict.values())))
sec_name_list = list(set(sec_name_list_dup))

sec_all_day = {}
for i in sec_name_list:
    sec_all_day[i] = pd.DataFrame()
    for j in day_list:
        flag = sec_dict_w_f[j].get(i)
        sec_all_day[i] = pd.concat([sec_all_day[i], flag])

factor_r_list = ["benchPrice", "midPrice", "bl_dist", "b_avl_dist", "ml_dist", "m_avl_dist",
                "ba_dist", "depth", "tVol_depth", "tVol_abVol",
                "rskew", "downward_ratio", "reverse", "pvol_corr","Return","Return_p_1"]

for i in sec_name_list:
    sec_all_day[i].to_csv("D:\\feature_cicc\\output_data\\"+i[2:-1]+".csv",index=False)

mean_df = {}
std_df = {}
for i in sec_all_day.keys():
    mean_df[i] = sec_all_day[i][factor_r_list].mean(numeric_only=True)
    std_df[i] = sec_all_day[i][factor_r_list].std(numeric_only=True)


for j in day_list:
    for i in security_name_dict[j]:
        for n in factor_r_list:
            sec_dict_w_f[j][i][n+"_z"] = (sec_dict_w_f[j][i][n]-mean_df[i][n])/std_df[i][n]


# get横截面data - 算IC
def get_interval_grp(time, interval_list):
    return np.searchsorted(interval_list, time, side='right')


all_data = pd.DataFrame()
for i in sec_name_list:
    for j in day_list:
        all_data = all_data.append(sec_dict_w_f[j].get(i),ignore_index=True)

starttime = min(all_data["trade_time"])
endtime = max(all_data["trade_time"])
window_len = dt.timedelta(milliseconds=3*1000)
interval_group = pd.date_range(start=starttime, end=endtime, freq=window_len).tolist()

all_data["interval_grp"] = np.searchsorted(interval_group, all_data["trade_time"], side='right')

factor_list_sigm = ["benchPrice_sigm", "midPrice_sigm", "bl_dist_sigm", "b_avl_dist_sigm", "ml_dist_sigm", "m_avl_dist_sigm",
                    "ba_dist_sigm", "depth_sigm","tVol_depth_sigm", "tVol_abVol_sigm", "rskew_sigm", "downward_ratio_sigm", "reverse_sigm",
                    "pvol_corr_sigm"]
factor_list_z = ["benchPrice_z", "midPrice_z", "bl_dist_z", "b_avl_dist_z", "ml_dist_z", "m_avl_dist_z",
                    "ba_dist_z", "depth_z","tVol_depth_z", "tVol_abVol_z", "rskew_z", "downward_ratio_z", "reverse_z",
                    "pvol_corr_z"]

all_data_group = all_data.groupby(["interval_grp"])
IC_sigm_df = pd.DataFrame(columns=factor_list_sigm)
rIC_sigm_df = pd.DataFrame(columns=factor_list_sigm)
IC_z_df = pd.DataFrame(columns=factor_list_z)
rIC_z_df = pd.DataFrame(columns=factor_list_z)


for factor_sigm in factor_list_sigm:
    f_temp_sigm = all_data_group[[factor_sigm,"Return_p_1_sigm"]]
    fr_corr_temp = f_temp_sigm.corr(method='pearson').unstack().iloc[:,1]
    fr_scorr_temp = f_temp_sigm.corr(method='spearman').unstack().iloc[:,1]
    pairs_num = f_temp_sigm.count().apply(min,axis=1)
    valid_group_name = list(pairs_num[pairs_num >= pairs_required].index)
    valid_fr_corr = fr_corr_temp[valid_group_name]
    valid_fr_scorr = fr_scorr_temp[valid_group_name]
    IC_sigm_df[factor_sigm] = valid_fr_corr
    rIC_sigm_df[factor_sigm] = valid_fr_scorr

for factor_z in factor_list_z:
    f_temp_z = all_data_group[[factor_z,"Return_p_1_z"]]
    fr_corr_temp_z = f_temp_z.corr(method='pearson').unstack().iloc[:,1]
    fr_scorr_temp_z = f_temp_z.corr(method='spearman').unstack().iloc[:,1]
    pairs_num_z = f_temp_z.count().apply(min,axis=1)
    valid_group_name_z = list(pairs_num_z[pairs_num_z >= pairs_required].index)
    valid_fr_corr_z = fr_corr_temp_z[valid_group_name_z]
    valid_fr_scorr_z = fr_scorr_temp_z[valid_group_name_z]
    IC_z_df[factor_z] = valid_fr_corr_z
    rIC_z_df[factor_z] = valid_fr_scorr_z

# IC_sigm_df.to_excel('D:/cicc-因子/因子研究/IC_sigm_f_5.xlsx')
# rIC_sigm_df.to_excel("D:/cicc-因子/因子研究/rIC_sigm_f_5.xlsx")
# IC_z_df.to_excel("D:/cicc-因子/因子研究/IC_z_f_5.xlsx")
# rIC_z_df.to_excel("D:/cicc-因子/因子研究/rIC_z_f_5.xlsx")

IC_sigm_df = pd.read_excel('D:/cicc-因子/因子研究/IC_sigm_f_10.xlsx',index_col=[0])
rIC_sigm_df = pd.read_excel("D:/cicc-因子/因子研究/rIC_sigm_f_10.xlsx",index_col=[0])
IC_z_df = pd.read_excel("D:/cicc-因子/因子研究/IC_z_f_10.xlsx",index_col=[0])
rIC_z_df = pd.read_excel("D:/cicc-因子/因子研究/rIC_z_f_10.xlsx",index_col=[0])


IC_sigm_cl = IC_sigm_df.dropna(how='all')
IC_mean_sigm = IC_sigm_cl.mean(skipna=True)
IC_std_sigm = IC_sigm_cl.std(skipna=True)
ICIR_sigm = IC_mean_sigm/IC_std_sigm
IC_pos_df_sigm = IC_sigm_cl>0
p_IC_pos_sigm = IC_pos_df_sigm.mean(skipna=True)

IC_z_cl = IC_z_df.dropna(how='all')
IC_mean_z = IC_z_cl.mean(skipna=True)
IC_std_z = IC_z_cl.std(skipna=True)
ICIR_z = IC_mean_z/IC_std_z
IC_pos_df_z = IC_z_cl>0
p_IC_pos_z = IC_pos_df_z.mean(skipna=True)

rIC_sigm_cl = rIC_sigm_df.dropna(how='all')
rIC_mean_sigm = rIC_sigm_cl.mean(skipna=True)
rIC_std_sigm = rIC_sigm_cl.std(skipna=True)
rICIR_sigm = rIC_mean_sigm/rIC_std_sigm
rIC_pos_df_sigm = rIC_sigm_cl>0
p_rIC_pos_sigm = rIC_pos_df_sigm.mean(skipna=True)

rIC_z_cl = rIC_z_df.dropna(how='all')
rIC_mean_z = rIC_z_cl.mean(skipna=True)
rIC_std_z = rIC_z_cl.std(skipna=True)
rICIR_z = rIC_mean_z/rIC_std_z
rIC_pos_df_z = rIC_z_cl>0
p_rIC_pos_z = rIC_pos_df_z.mean(skipna=True)

result_table_sigm = pd.DataFrame(columns = ["IC_mean","ICIR","p(IC>0)",
                                       "rIC_mean","rICIR","p(rIC>0)"])
result_table_sigm["IC_mean"] = IC_mean_sigm
result_table_sigm["ICIR"] = ICIR_sigm
result_table_sigm["p(IC>0)"] = p_IC_pos_sigm
result_table_sigm["rIC_mean"] = rIC_mean_sigm
result_table_sigm["rICIR"] = rICIR_sigm
result_table_sigm["p(rIC>0)"] = p_rIC_pos_sigm

result_table_z = pd.DataFrame(columns = ["IC_mean","ICIR","p(IC>0)",
                                       "rIC_mean","rICIR","p(rIC>0)"])
result_table_z["IC_mean"] = IC_mean_z
result_table_z["ICIR"] = ICIR_z
result_table_z["p(IC>0)"] = p_IC_pos_z
result_table_z["rIC_mean"] = rIC_mean_z
result_table_z["rICIR"] = rICIR_z
result_table_z["p(rIC>0)"] = p_rIC_pos_z

result_table_sigm.to_csv('D:/cicc-因子/因子研究/result_table_sigm10.csv')
result_table_z.to_csv('D:/cicc-因子/因子研究/result_table_z10.csv')



