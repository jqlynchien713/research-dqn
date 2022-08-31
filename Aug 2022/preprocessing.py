# Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import itertools

DATA = pd.read_csv('../data/6897-1y-c.csv')

## Preprocessing
### Handle Username Formats

def replace_invalid(string):
  string = string.replace('-', '')
  string = string.replace('/', '')
  string = string.replace('*', '')
  string = string.replace(' ', '')
  string = string.replace('劉德玲', '')
  return string

DATA['聯絡電話'] = DATA['聯絡電話'].apply(replace_invalid)

### 處理 period （前六碼）

DATA['period'] = DATA['下單日期'].astype(str).apply(lambda x: x[:6])

### One Hot Encoding
# * 處理 category, shipment, payment
# * 要先做 label encoding 才能做 one hot encoding

le = LabelEncoder()
DATA['category_label'] = le.fit_transform(DATA['商品分類'])
DATA = pd.concat([DATA, pd.get_dummies(DATA['category_label'], prefix='cat')], axis = 1)

DATA['shipment'] = le.fit_transform(DATA['運送方式'])
DATA['payment'] = le.fit_transform(DATA['付款方式'])
DATA = pd.get_dummies(DATA,
                      prefix=['shipment', 'payment'],
                      columns=['shipment', 'payment'])

### Create `item_id`

# Generating item ids
shuffled_items = DATA['商品名稱'].sample(frac=1).reset_index(drop=True).unique()
item_dict = { x: i for i, x in enumerate(shuffled_items) }

DATA['item_id'] = DATA['商品名稱'].map(item_dict)

DATA['場次'] = DATA['場次'].fillna(DATA['下單日期'])

## Constants Needed

LB_CE = [f'cat_{i}' for i in range(228)] + [f'shipment_{i}' for i in range(4)] + [f'payment_{i}' for i in range(2)]
USER_LIST = DATA['聯絡電話'].unique()
STREAM_LIST = DATA['場次'].unique()
STREAM_LIST.sort()

# Creating the row axis labels 
LB_PERIOD = list(DATA['period'].unique())
LB_USER = ['user']
LB_PQ = ['total_price', 'total_quantity']
LB_CAT_DATA = [f'cat_{x}' for x in list(range(228))]
LB_CAT_ITEM = [f'i_cat_{x}' for x in list(range(228))]
LB_SHIPMENT_PAYMENT = ['shipment_0', 'shipment_1', 'shipment_2', 'shipment_3', 'payment_0', 'payment_1']
USER_LB = LB_PERIOD + LB_USER + LB_PQ + LB_SHIPMENT_PAYMENT
LB_ITEMS = ['i_item_id', 'i_avg_price', 'i_count'] + LB_CAT_ITEM


## Context Representation

USER_STREAM_CONTEXT = DATA.groupby(['聯絡電話', '場次']).sum().loc[:, :'payment_1']

USER_STREAM_CONTEXT['user'] = USER_STREAM_CONTEXT.apply(lambda x: x.name[0], axis=1)

## Item Representation
# - 商品 id 來源為 `item_dict`: `商品名稱: item_id`
# - 可用欄位： `categories(one-hot)`, `price`, `被購買次數`
# - Columns: `['下單日期', '商品名稱', '規格', '單價', '數量', '折扣', '總金額', '專屬折扣', '運費', '信用卡手續費', '紅利折抵', '收款金額', '付款方式', '運送方式', '收件人', '寄送地址', '聯絡電話', '場次', '處理後名稱', '商品分類', 'period', 'category_label', 0-227, 'shipment_0', 'shipment_1', 'shipment_2', 'shipment_3', 'payment_0', 'payment_1', 'item_id']`

def get_item_df():
  item_df = pd.DataFrame(columns=LB_ITEMS)
  item_df['i_item_id'] = DATA.item_id.unique()
  # Count
  item_count = DATA.groupby('item_id').size()
  item_df['i_count'] = item_df['i_item_id'].apply(lambda x: item_count[x])
  # Cat
  item_df[LB_CAT_ITEM] = DATA.groupby('item_id').sum()[LB_CAT_DATA]
  # Price
  item_df['i_avg_price'] = DATA.groupby('item_id').mean()['單價']
  return item_df

ITEM_DF = get_item_df()

# Generate Item x Stream data
# * unique items x each stream
# * index: item_id
# * column: stream_id

def map_item_stream(item_id):
  in_streams = DATA.loc[DATA.item_id == item_id]['場次'].unique()
  res_series = pd.Series([0]*len(STREAM_LIST))
  res_series.index = STREAM_LIST
  res_series[in_streams] = 1
  return res_series

ITEM_STREAM_DF = pd.DataFrame(index=DATA.item_id.unique(), columns=DATA['場次'].unique())
ITEM_STREAM_DF = ITEM_STREAM_DF.apply(lambda x: map_item_stream(x.name), axis=1, result_type='expand')

## [Reward] Generate Real Bought DF

REAL_BOUGHT_DF = DATA.loc[:, ['聯絡電話', '場次', 'item_id']]

## [Action] User bought last stream

LAST_BOUGHT_STREAM = USER_STREAM_CONTEXT.reset_index().groupby('聯絡電話', as_index=False).last().loc[:, ['聯絡電話', '場次']].set_index('聯絡電話')

# USER_STREAM_CONTEXT, ITEM_DF, ITEM_STREAM_DF, REAL_BOUGHT_DF, LAST_BOUGHT_STREAM, LB_ITEMS
