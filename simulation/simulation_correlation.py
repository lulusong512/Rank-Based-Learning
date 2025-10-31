# Import required libraries
import numpy as np
import pandas as pd

def add_correlation(random_seed, case_df, control_df, significant_feature_indices):
    """
    Add correlation among significant features in simulated case-control data.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    case_df : pd.DataFrame
        Simulated case data from `simulation_data()`.
    control_df : pd.DataFrame
        Simulated control data from `simulation_data()`.
    significant_feature_indices : list or np.ndarray
        Indices of features that are significantly related to the outcome.

    Returns
    -------
    case_data : pd.DataFrame
        Modified case data with induced correlation among significant features.
    df : pd.DataFrame
        Combined dataset (cases with correlation + original controls).

    Notes
    -----
    - For all significant features, the same base signal is added to create correlation.
    - Non-significant features remain unchanged.
    - The combined dataset maintains the same structure as the original simulation output.
    """

    # Total number of features in dataset
    num_features = 300
    
    # Create a copy of the case data to avoid altering the original
    case_data = case_df.copy()

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Loop through each feature
    for j in range(num_features):
        # Only modify significant features
        if j in significant_feature_indices:
            # Generate base shared signal across significant features
            first_feature_data = np.random.normal(loc=10, scale=10, size=len(case_df))
            
            # Add correlation by adding shared signal to all significant features
            for index in significant_feature_indices:
                case_data.iloc[:, index] += first_feature_data
                
    # Combine modified case data and unmodified control data
    df = pd.concat([case_data, control_df], axis=0)

    return case_data, df
