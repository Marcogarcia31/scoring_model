from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from Feature_selector import FeatureSelector
import numpy as np


class Selector_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        just_to_add_a_line = 42
    
    def fit(self, X, y = None):

      X = X.copy()
      
      fs = FeatureSelector(data = X, labels = y)
      
      fs.identify_missing(missing_threshold = 0.6)
      
      fs.identify_single_unique()
      
      fs.identify_collinear(correlation_threshold = 0.98)

      features_to_remove = fs.ops['collinear']
      features_to_remove.extend(fs.ops['missing'])
      features_to_remove.extend(fs.ops['single_unique'])

      features_to_remove = list(set(features_to_remove))

      X = X.drop(features_to_remove, axis= 1)

      
      ### Compute feature importances
      fs = FeatureSelector(data = X, labels = y)

      
      fs.identify_zero_importance(task = 'classification', 
                            eval_metric = 'auc', 
                        n_iterations = 10, 
                             early_stopping = True)
      
      fs.identify_low_importance(cumulative_importance = 0.99)

      new_features_to_remove = fs.ops['zero_importance']

      new_features_to_remove.extend(fs.ops['low_importance'])
      
      new_features_to_remove = list(set(new_features_to_remove))

      features_to_remove.extend(new_features_to_remove)
      
      self.features_to_remove = features_to_remove

      return self


    
    def transform(self, X, y = None):

      ### Drop columns
      X = X.copy()
      X = X.drop(self.features_to_remove, axis = 1)
      
      return X




class Features_generator(BaseEstimator, TransformerMixin):
    def __init__(self):
        just_to_add_a_line = 42
    
    def fit(self, X, y = None):
      return self
    
    def transform(self, X, y = None):

      X = X.copy()

      X['ANNUITY_on_INCOME_ratio'] = np.where(X['AMT_ANNUITY_x'].notnull() & X['AMT_INCOME_TOTAL'].notnull(), X['AMT_ANNUITY_x']/X['AMT_INCOME_TOTAL'], np.nan)
      X['ANNUITY_on_CREDIT_ratio'] = np.where(X['AMT_ANNUITY_x'].notnull() & X['AMT_CREDIT_x'].notnull(), X['AMT_ANNUITY_x']/X['AMT_CREDIT_x'], np.nan)
      X['INCOME_on_CREDIT_ratio'] = np.where(X['AMT_INCOME_TOTAL'].notnull() & X['AMT_CREDIT_x'].notnull(), X['AMT_INCOME_TOTAL']/X['AMT_CREDIT_x'], np.nan)
      
      return X