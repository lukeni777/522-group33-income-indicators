import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate_model import cross_val_model_results

X_sample = pd.DataFrame({'X1': np.random.randint(3, 20, 10), 
                         'X2': np.random.randint(5, 30, 10), 
                         'X3': np.random.randint(15, 50, 10)})
y_sample = np.ones(10)
def test_cross_val_model_results_single():
    """
    Test the cross_val_model_results function for correct output structure.
    
    This test verifies that the cross_val_model_results function returns a 
    properly formatted DataFrame with the expected structure when provided with
    a basic estimator and sample data.
    
    Test Conditions
    ---------------
    - Uses LinearRegression as a sample estimator
    - Uses pre-defined sample data (X_sample, y_sample)
    - Uses 'accuracy' as the scoring metric
    """
    out_sample_single = cross_val_model_results(estimator=LinearRegression(), 
                                         X_train=X_sample,
                                         y_train=y_sample,
                                         scoring='accuracy')
    
    assert isinstance(out_sample_single, pd.DataFrame)
    assert len(out_sample_single.columns.to_list()) == 2 
