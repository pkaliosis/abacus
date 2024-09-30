# Define a function to calculate MAE and RMSE
def mae(n_objects_col, predicted_counts_col):
    
    # Calculate MAE
    return np.mean(np.abs(n_objects_col - predicted_counts_col))
    
# Define a function to calculate MAE and RMSE
def rmse(n_objects_col, predicted_counts_col):
    
    # Calculate RMSE
    return np.sqrt(np.mean((n_objects_col - predicted_counts_col) ** 2))

import pandas as pd
import numpy as np
test_df = pd.read_csv("/home/ubuntu/pkaliosis/zsoc/outputs/dfs/test_df_pred.csv")
# Evaluation
mae = mae(test_df["n_objects"], test_df["predicted_counts"])
rmse = rmse(test_df["n_objects"], test_df["predicted_counts"])
        
print("MAE:", mae)
print("RMSE:", rmse)