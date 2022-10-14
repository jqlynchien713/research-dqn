### Collecting Training Data
import pandas as pd
import numpy as np
from utils import gen_exist_series, get_full_state, get_reward, calculate_interest_change, get_input, model_predict_top10, INPUT_DF_COL

class ReplayBuffer:
  def __init__(self, max_memory=100000, discount=.9):
    """
    Setup
    max_memory: the maximum number of experiences we want to store
    memory: a list of experiences
    discount: the discount factor for future experience
    In the memory the information whether the game ended at the state is stored seperately in a nested array
    [...
    [experience, game_over]
    [experience, game_over]
    ...]
    """
    self.max_memory = max_memory
    self.memory = list()
    self.discount = discount

  def remember(self, interest_score, states, game_over):
    # Save a state to memory
    self.memory.append([interest_score, states, game_over])
    # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
    if len(self.memory) > self.max_memory:
      del self.memory[0]

  def get_batch(self, model, batch_size=10):
    # How many experiences do we have?
    len_memory = len(self.memory)

    # Calculate the number of actions that can possibly be taken in the game.
    # Actions: 0 = not recommend, 1 = recommend
    num_actions = model.output_shape[-1]

    # Dimensions of our observed states, ie, the input to our model.
    # Memory:  [
    #   [interest_score, [ [...state], action, reward, next_state_idx], game_over],
    #   [interest_score, [ [...state], action, reward, nexr_state_idx], game_over],
    #   ...
    # ]
    env_dim = len(INPUT_DF_COL)

    inputs = pd.DataFrame(columns=INPUT_DF_COL)
    targets = pd.DataFrame(columns=[0])
    
    
    
    # We draw states to learn from randomly
    for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):      

      # Here we load one transition <s, a, r, s'> from memory
      state_t, action_t, reward_t, state_tp1 = self.memory[idx][1]
      # state_t = state_t.astype('float32')
      game_over = self.memory[idx][2]

      '''
      修改倒入 state 的方式 input = (state - item) + item_feat
      拆掉 model_predict 成 function
      
      here should be state_t * all_items
      '''
      state_t = get_input(state_t).astype('float32')
      # puts state into input
      inputs = pd.concat([inputs, state_t], axis=0)
      # TODO: Modify input shape 0

      # First we fill the target values with the predictions of the model.
      # They will not be affected by training (since the training loss for them is 0)
      # TODO
      '''
      每個 actions 都會被 predict 一個成績/reward
      '''

      # if the game ended, the reward is the final reward
      if game_over:  # if game_over is True
        state_t['reward'] = reward_t
        targets = pd.concat([targets, reward_t], axis=0).astype('float32')
      else:
        state_t['reward'] = model.predict(state_t).flatten()
        # 找到 action_t 們，指到 state_t 上去算 discount values
        action_t = action_t[action_t == 1].index.to_list()
        
        state_tp1 = get_input(state_tp1)
        Q_sa = np.max(model.predict(state_tp1)[0])
        # r + gamma * max Q(s',a')
        # DataFrame apply
        state_t.loc[action_t, 'reward'] = state_t.loc[action_t, 'reward'].apply(lambda x: x + self.discount * Q_sa).astype('float32')
        targets = pd.concat([targets, state_t['reward']], axis=0).astype('float32')
    return inputs, targets
