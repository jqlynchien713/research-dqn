# Utilities
# import from preprocessing
import pandas as pd
from preprocessing import USER_STREAM_CONTEXT, ITEM_DF, ITEM_STREAM_DF, REAL_BOUGHT_DF, LAST_BOUGHT_STREAM, LB_ITEMS, USER_LIST, LB_CE, STREAM_LIST

### Environment
# Get full state

# * 參考原論文: user _interest 初始值為 random vector
# 1. 很多筆記錄裡用戶只在一場直播裡購買，如果 random 的話 accuracy 會下降
# 2. 不可能用該期直播的相關紀錄作為 state → 在 user/state context 裡加上 RFML/streamer information
# 3. user x all_stream → 在第一場直播都給予 random vector，對 accuracy 的影響就不會太高（？


'''
Generate series: whether elements in A existed in list B
A, B: List
return: pd.Series
example:
  A: [1, 2, 4, 5]
  B: [1, 2, 3, 4, 5, 6, 7]
  return: Series([1, 1, 0, 1, 1, 0, 0], index=[1, 2, 3, 4, 5, 6, 7])
'''
def gen_exist_series(A, B):
  exist_list = [int(item in A) for item in B]
  return pd.Series(exist_list, index=B)

USER_ALL_STREAM_INIT = USER_STREAM_CONTEXT.describe().loc['50%']

def get_full_state(user_all_streams, stream_list, i):
  # Get full state: current_state = user_stream + item_stream
  # 第一次參加直播/cold start
  # CE paper: user_interest part init with random vector
  # TODO init with random values
  #      Cold start problem
  # USER PART
  user_part = USER_ALL_STREAM_INIT.copy() if (i - 1) == -1 else user_all_streams.loc[stream_list[(i - 1)]]
  user_part['user'] = user_all_streams['user'].to_list()[0]

  # ITEMS PART
  # Get all items from stream
  # USER_PART + ITEM_PART
  full_state_stream_items = ITEM_STREAM_DF.loc[ITEM_STREAM_DF[stream_list[i]] == 1].index.to_list()
  user_part['cand_item'] = full_state_stream_items
  return user_part


'''
Comparison function for reward
'''
def r(a, b):
  if a==1 and b==0: return 0 # -1 when the rule is to punish unrec-bought
  else: return a & b

def get_reward(user_phone, stream, action_ids):
  items = action_ids.index
  real_bought_ids = REAL_BOUGHT_DF.loc[(REAL_BOUGHT_DF['聯絡電話'] == user_phone) 
                                         & (REAL_BOUGHT_DF['場次'] == stream)]['item_id'].values
  real_bought_ids_series = gen_exist_series(real_bought_ids, items)
  
  reward_list = [r(a, b) for a, b in zip(real_bought_ids_series.values, action_ids.values)]
  return pd.Series(reward_list, index=items)


from sklearn.metrics import log_loss

def calculate_interest_change(user_all_streams, stream_list, i):
  if i < 0:
    return 0
  else:
    former_stream = stream_list[i-1]
    current_stream = stream_list[i]

    test_ce = user_all_streams.loc[:, LB_CE]
    ce = log_loss(test_ce.loc[former_stream], test_ce.loc[current_stream], labels=['prev', 'current'])
    if ce < 0.01: ce = 0
    else: ce = round(ce, 3)
    return ce

INPUT_DF_COL__USR = USER_STREAM_CONTEXT.columns.to_list()
INPUT_DF_COL = INPUT_DF_COL__USR + LB_ITEMS

'''
Convert state format to model input format
'''
def get_input(input_state):
  # Slice items
  items = input_state['cand_item']
  input_state = input_state.drop('cand_item')
  item_feat = ITEM_DF.loc[items]
  
  # Create new dataframe
  stream_item_feat = pd.DataFrame(columns=INPUT_DF_COL, index=item_feat.index)
  
  # Fill in other context
  # stream_item_feat.loc[:, INPUT_DF_COL__USR].assign(**input_state)
  stream_item_feat = stream_item_feat.loc[:, INPUT_DF_COL__USR].assign(**input_state)
  
  # Fill in items
  stream_item_feat[LB_ITEMS] = item_feat
  return stream_item_feat.astype('float32')

def model_predict_top10(model, input_state):
  # Get all items
  full_input = get_input(input_state).astype('float32')
  
  # 紀錄所有預測結果
  predicts = model.predict(full_input)
  full_input['predict'] = predicts
  actions = full_input['predict'].nlargest(10).index #['i_item_id'].to_list()
  actions = full_input.loc[actions, 'i_item_id'].values
  return actions


# Export list: gen_exist_series, get_full_state, get_reward, calculate_interest_change, get_input, model_predict_top10