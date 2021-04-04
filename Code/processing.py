import os
import json
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from dataclasses import dataclass

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# List of swing states in the 2016 election
swing_states = ['WI', 'PA', 'NH', 'MN', 'AZ', 'GA', 'VA', 'FL', 'MU', 'NV', 'CI', 'NC', 'ME']

@dataclass
class Dataset:
  expense_path : str
  candidate_path : str
  categories : dict

  cols = ['CMTE_ID', 'RPT_YR', 'STATE', 'TRANSACTION_DT', 
          'TRANSACTION_AMT', 'PURPOSE']

  def __post_init__(self):
    # Reading in data
    self.expenses = pd.read_csv(self.expense_path)
    self.candidates = pd.read_csv(self.candidate_path)

    # Cleaning and subsetting 
    # There are a few incorrectly inputted dates so those are dropped
    self.expenses = self.expenses[self.cols].dropna()
    self.expenses = self.expenses.query("TRANSACTION_AMT > 0 ")
    self.expenses['dummy_year'] = self.expenses['TRANSACTION_DT'].apply(lambda x : x.split('/')[-1]).astype(int)
    self.expenses.loc[:,'RPT_YR'] =  self.expenses['RPT_YR'].astype(int)
    min_year, max_year = self.expenses['RPT_YR'].min(),  self.expenses['RPT_YR'].max()
    self.expenses = self.expenses.query("dummy_year >= @min_year & dummy_year <= @max_year")
    self.expenses['datetime'] = pd.to_datetime(self.expenses['TRANSACTION_DT'], format= '%m/%d/%Y')
    self.expenses.loc[:, 'PURPOSE'] = self.expenses['PURPOSE'].apply(lambda x: self.clean_text(x))
    self.expenses.drop(columns = ['dummy_year', 'TRANSACTION_DT'], inplace = True)

    # formattng last name
    self.candidates['lastname'] = self.candidates['CAND_NAME'].apply(lambda x: x.split(' ')[-1])

    # useful for applying category mapping on list of transaction types
    inverted_categories = {v: key for key, value in self.categories.items() for v in value}
    self.expenses.loc[:, 'PURPOSE'] = self.expenses['PURPOSE'].map(inverted_categories)
    _cols = [i for i in self.expenses.columns if 'PURPOSE_' in i]
    self.expenses.loc[:, _cols] = self.expenses[_cols].multiply(
        self.expenses['TRANSACTION_AMT'].values, axis = 0)

  def clean_text(self, text):
    # Reformats text to be all lowercase, have no punctuation, and be equally spaced
    for char in string.punctuation:
        text = text.replace(char,' ').lower()
    text_list = [x.strip() for x in text.split()]
    return ' '.join(text_list)

  def query(self, params):
    """
    input: type(params) = dict
    example params = {'year': 2004, 'candidates': ['Bush', 'Gore']}
    """
    # setting year range
    min_year, max_year = params['year']-4, params['year']
    cands = params['candidates']
    # selecting candidates for the input election year range
    _cols = ['CAND_NAME', 'CAND_PCC', 'CAND_PTY_AFFILIATION']
    cands = self.candidates.query("CAND_ELECTION_YR > @min_year & CAND_ELECTION_YR <= @max_year & lastname in @cands")

    # subsetting data and mergin candidate meta data (committee ID and candidate name etc)
    expense_df = self.expenses.query("CMTE_ID in @cands.CAND_PCC & RPT_YR > @min_year & RPT_YR <= @max_year")
    expense_df = expense_df.merge(cands[_cols], left_on = 'CMTE_ID', right_on = 'CAND_PCC')
    return expense_df

def get_covariates(expenses, demographics):
  """  
  Ugliest piece of code ever but i dont care!
  To-Do : Make it to where I can provide a list of dates with which to partitition
          the data to have q1, q2, q3, q4. Effectively len(X.columns)*4.
  """  
  # hot encoding and scaling by transaction amount (useful for groupby)
  df = pd.get_dummies(expenses, columns=['PURPOSE'])
  cols = [i for i in df.columns if 'PURPOSE_' in i]
  party = ['CAND_PTY_AFFILIATION']
  df.loc[:,cols] = df[cols].multiply(df['TRANSACTION_AMT'], axis = 0)

  # grouping by state party and pivoting + cleaning up
  X = df.groupby(['STATE','CAND_PTY_AFFILIATION'])[cols].sum().reset_index()
  X = pd.pivot(X, columns = party).fillna(0)
  X.columns =   [' '.join(col).strip() for col in X.columns.values]
  X['STATE'] = (X['STATE DEM'].astype(str)+X['STATE REP'].astype(str)).apply(lambda x:x.strip('0'))
  X.drop(columns = ['STATE DEM', 'STATE REP'], inplace = True)
  X = X.groupby('STATE').sum().reset_index().query('STATE in @states')

  # merging demographics and sorting columns so that they are consistent across all models
  X = X.merge(demographics, left_on='STATE', right_on='state').drop(columns = 'state')
  X.rename(columns = {col:' '.join(col.split(' ')[::-1]).replace('PURPOSE_','') for col in X.columns}, inplace=True)
  X = X[['STATE'] + sorted([i for i in X.columns if i != 'STATE'])]
  X.index = X['STATE']
  X.drop(columns=['STATE']+[cols for cols in X.columns 
    if 'other' in cols and cols != 'hisp_other_pct'], inplace = True)
  return X.sort_index()

def get_expenses(data, params):
  # specifying the election by year
  expenses = data.query(params)
  expenses['lastname'] = expenses['CAND_NAME'].apply(lambda x: x.split(' ')[-1])
  return expenses