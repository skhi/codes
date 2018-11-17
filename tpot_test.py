#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:30:54 2018

@author: nikoloz.s
"""

import os
os.chdir('/Users/nikoloz.s/Documents/DataScience/SSMM')
import math
import sys
import importlib

import numpy as np

import pandas as pd

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from scipy.stats import norm
import matplotlib.pyplot as plt


from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


import scikitplot as skplt
import catboost as cb


from tpot import TPOTClassifier

best25_vars = [
'cst_e_t_ds_features_bpno_acc_created_days_churn_imputed',
'cst_e_t_ds_features_bpno_mob_reg_ge1y_le3y_imputed',
'cst_e_t_ds_features_bpno_launch_days_churn_imputed',
'cst_e_t_ds_features_bpno_number_non_mobile_products_registered_imputed',
'cst_e_t_ds_features_bpno_email_join_days_churn_imputed',
'cst_e_t_ds_features_bpno_spay_user_churn_indexed',
'cst_e_t_ds_features_bpno_gender_male_indexed',
'cst_e_t_ds_features_bpno_registered_days_churn_imputed',
'cst_e_t_ds_features_bpno_income_median_imputed',
'cst_e_t_ds_features_bpno_number_previously_registered_premium_tvs_imputed',
'cst_e_t_ds_features_bpno_dvc_count_churn_imputed',
'cst_e_t_ds_features_bpno_user_type_churn_indexed',
'cst_e_t_ds_features_bpno_carrier_churn_indexed',
'cst_e_t_ds_features_bpno_other_reg_ge1y_le3y_imputed',
'cst_e_t_ds_features_bpno_mob_reg_ge180_le_1y_imputed',
'cst_e_t_ds_features_bpno_gender_female_indexed',
'cst_e_t_ds_features_bpno_prem_segment_indexed',
'cst_e_t_ds_features_bpno_number_previously_registered_tvs_imputed',
'cst_e_t_ds_features_bpno_mob_reg_ge_3y_imputed',
'cst_e_t_ds_features_bpno_mob_reg_le180days_imputed',
'cst_e_t_ds_features_bpno_number_computing_products_registered_imputed',
'cst_e_t_ds_features_bpno_number_registered_full_hd_tvs_imputed',
'cst_e_t_ds_features_bpno_number_home_appliances_products_registered_imputed',
'cst_e_t_ds_features_bpno_premtv_model_group_indexed',
'cst_e_t_ds_features_bpno_home_pct_own_imputed',
'ecom_purchased_indexed'
]


### read data
data_path = r'/Users/nikoloz.s/Documents/Data/data.csv'


data = pd.DataFrame()
tmp = pd.read_csv(data_path,
                  sep=',', 
                  low_memory=False,
                  iterator=True,
                  chunksize=500000)
data = pd.concat(tmp, ignore_index=True)


###data = pd.read_csv(data_path, nrows=500)

data = data.drop('Unnamed: 0', axis=1)

data = data[best25_vars]

cat_cols= [col for col in data.columns.tolist() if 'indexed' in col and 'ecom_purchased_indexed' not in col]
categorical_features_indices = [data.columns.get_loc(c) for c in cat_cols]
for col in cat_cols:
    data[col]=data[col].astype("O")
    
data=pd.get_dummies(data, columns= cat_cols)
    
X_train, X_test, y_train, y_test = train_test_split(data.drop('ecom_purchased_indexed', axis=1), 
                                                    data['ecom_purchased_indexed'],
                                                    random_state=42,
                                                    train_size=0.60, 
                                                    test_size=0.40)


import time

start = time.time()
tpot = TPOTClassifier(generations=6, verbosity=2, max_eval_time_mins=10, population_size=40,scoring='f1_weighted')
tpot.fit(X_train.values, y_train.values)
end = time.time()

print ("Time needed to run pipelines is {}".format(end-start))

print (tpot.score(X_test.values, y_test.values))
tpot.export('tpot_titanic_pipeline_fulldata.py')


