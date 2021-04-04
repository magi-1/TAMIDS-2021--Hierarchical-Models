import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def map_colors(state_list):
  swing_states = ['WI', 'PA', 'NH', 'MN', 'AZ', 'GA', 'VA', 'FL', 'MU', 'NV', 'CI', 'NC', 'ME']
  return ['tab:blue' if s in swing_states else 'k' for s in state_list]

def inspect_samples(coefs, states, col_names, N, M):
  # N = # of bars for ax1
  # M = # of bars for ax2

  states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
            
  tot_variance = np.mean(np.var(coefs, axis = 0), axis = 1)
  coef_mu = np.mean(np.mean(coefs, axis = 0), axis = 0)

  var_pairs = [(states[i], tot_variance[i]) for i in range(51)]
  var_pairs.sort(key = lambda x: x[1], reverse = True)
  var_ticknames, var_height = [x[0] for x in var_pairs[:N]], [x[1] for x in var_pairs[:N]]
  mu_pairs = [(col_names[i], coef_mu[i]) for i in range(18)]
  mu_pairs.sort(key = lambda x: x[1], reverse = True)
  mu_ticknames, mu_height = [x[0] for x in mu_pairs[:M]], [x[1] for x in mu_pairs[:M]]

  fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,11))
  ax1xticks = [(i+.5)*2 for i in np.array(list(range(N)))]
  ax2xticks = np.array(list(range(M)))
  ax1.bar(ax1xticks, height=var_height, color = map_colors(var_ticknames), width = 1.33)
  ax1.set_title('Param Uncertainty | (Blue = Swing State)')
  ax1.set_ylabel('mean(var(samples))')
  ax1.set_xticks(ax1xticks)
  ax1.set_xticklabels(var_ticknames)
  ax2.bar(ax2xticks, height=mu_height, color = map_colors(mu_ticknames))
  ax2.set_title('Avg Size of Covariates')
  ax2.set_ylabel('mean(mean(samples))')
  ax2.set_xticks(ax2xticks)
  ax2.set_xticklabels(mu_ticknames, rotation = 90)
  fig.savefig('hierarchical_summary.png', dpi=fig.dpi)
  plt.show()

def pickle_model(data_dict, path):
  # {'model': Induv_Model, 'trace': Induv_Trace, 'X_shared': X, 'lambdas': {'lambda1': Lambda1, 'Lambda2': Lambda2}}
  with open(path, 'wb') as buff:
      pickle.dump(data_dict, buff)

def unpickle_model(model_path):
  with open(model_path, 'rb') as input:
    model_dict = pickle.load(input)
  return model_dict