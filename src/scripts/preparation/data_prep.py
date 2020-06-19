# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import seaborn as sns
import datasist.project as dp
import datasist as ds

#read data from the raw data directory using datasist
data = dp.get_data('train.csv', loc='raw', method='csv')
ds.structdata.describe(data)

# +
#check for missing values
ds.structdata.display_missing(data)

#seperate the label from the data
label = data.Rating
data.drop(columns=['Rating'], inplace=True)


#Encode all categorical feature with label encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for col in data.columns:
    data[col] = lb.fit_transform(data[col])
    
    
data.head()
# -

#export the processed data and label to the processed folder
dp.save_data(data, 'train_proc')
dp.save_data(label, 'train_labels')


