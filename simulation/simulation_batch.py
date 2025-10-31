# Import required libraries
import numpy as np
import pandas as pd


def simulation_data_with_batch_effect(random_seed: int, num_significant_features: int):
    """
    Generate simulated case-control data with batch effects (enlarged dataset scenario).

    This function extends the true differential simulation by introducing artificial
    batch effects across different sample groups to mimic real-world variability such
    as technical differences across batches or experimental runs.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    num_significant_features : int
        Number of features significantly associated with the outcome.

    Returns
    -------
    case_df : pd.DataFrame
        Original simulated case data (n=30).
    control_df : pd.DataFrame
        Original simulated control data (n=30).
    df_ori : pd.DataFrame
        Combined original case-control dataset (60 samples total).
    case_df_enlarged : pd.DataFrame
        Case samples after applying batch effect.
    control_df_enlarged : pd.DataFrame
        Control samples after applying batch effect.
    df_enlarged : pd.DataFrame
        Full batch-affected dataset combining multiple enlarged case/control sets.

    """
    
    # Set seed for reproducibility
    np.random.seed(random_seed)

    # Define the number of features, cases, and controls
    num_cases = 30
    num_controls = 30
    num_features = 300

    # Produce case and control data
    case_data = np.zeros((num_cases, num_features))
    control_data = np.zeros((num_controls, num_features))

    # Generate indices of significant features related to the outcome
    significant_feature_indices = np.random.choice(num_features, num_significant_features, replace=False)

    # Generate data for each feature for cases and controls
    for j in range(num_features):
        # Generate different normal distribution parameters for each feature
        mean_case = np.random.uniform(50, 70)
        std_case = np.random.uniform(1, 80)
        mean_control = np.random.uniform(50, 70)  
        std_control = np.random.uniform(1, 80)

        # Generate data for cases and controls
        case_data[:, j] = np.random.normal(loc=mean_case, scale=std_case, size=num_cases)
        control_data[:, j] = np.random.normal(loc=mean_control, scale=std_control, size=num_controls)
        
        # For features significantly related to the outcome, add additional noise to enhance correlation
        if j in significant_feature_indices:
            case_data[:, j] += np.random.normal(loc=60, scale=60, size=num_cases)

    # Create DataFrame for cases and controls
    case_df = pd.DataFrame(case_data, columns=[f'feature_{i}' for i in range(num_features)])
    control_df = pd.DataFrame(control_data, columns=[f'feature_{i}' for i in range(num_features)])
    
    case_df[case_df< 0] = 0
    control_df[control_df< 0] = 0
    
    # Add batch effect to create enlarged dataset
    batch_effect = np.random.uniform(100, 300)  # Broadcasting to match shape of case_df and control_df
    case_df_enlarged = case_df.iloc[:10, ] + batch_effect
    control_df_enlarged = control_df.iloc[:10, ] + batch_effect
    
    batch_effect1 = np.random.uniform(300, 500)  # Broadcasting to match shape of case_df and control_df
    case_df_enlarged1 = case_df.iloc[10:20, ] + batch_effect1
    control_df_enlarged1 = control_df.iloc[10:20, ] + batch_effect1
    
    batch_effect2 = np.random.uniform(500, 700)  # Broadcasting to match shape of case_df and control_df
    case_df_enlarged2 = case_df.iloc[20:30, ] + batch_effect2
    control_df_enlarged2 = control_df.iloc[20:30, ] + batch_effect2
    
    case_dfs = [case_df, case_df_enlarged, case_df_enlarged1, case_df_enlarged2]
    control_dfs = [control_df, control_df_enlarged, control_df_enlarged1, control_df_enlarged2]

    for df in case_dfs:
        df["cancer"] = 1

    for df in control_dfs:
        df["cancer"] = 0
    
    # Concatenate case and control datasets
    df_ori = pd.concat([case_df, control_df], ignore_index=True)
    df_enlarged = pd.concat([case_df_enlarged,case_df_enlarged1,case_df_enlarged2, control_df_enlarged,control_df_enlarged1,control_df_enlarged2], ignore_index=True)
    
    return case_df,control_df, df_ori,case_df_enlarged, control_df_enlarged, df_enlarged
