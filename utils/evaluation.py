import numpy as np

# Define a function to calculate MAE and RMSE
def mae(n_objects_col, predicted_counts_col):
    
    # Calculate MAE
    return np.mean(np.abs(n_objects_col - predicted_counts_col))
    
# Define a function to calculate MAE and RMSE
def rmse(n_objects_col, predicted_counts_col):
    
    # Calculate RMSE
    return np.sqrt(np.mean((n_objects_col - predicted_counts_col) ** 2))

        