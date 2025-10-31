import numpy as np
import pandas as pd

def simulation_data_true_differeial_60_samples(random_seed: int, num_significant_features: int):
    """
    Generate simulated case-control data with significant and non-significant features.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    num_significant_features : int
        Number of features significantly associated with the outcome.

    Returns
    -------
    case_df : pd.DataFrame
        Simulated data for case group.
    control_df : pd.DataFrame
        Simulated data for control group.
    simulated_data : pd.DataFrame
        Combined dataset with a 'cancer' outcome column (1=case, 0=control).
    """

    np.random.seed(random_seed)

    num_cases = 30
    num_controls = 30
    num_features = 300

    case_data = np.zeros((num_cases, num_features))
    control_data = np.zeros((num_controls, num_features))

    # Randomly select indices of significant features
    significant_feature_indices = np.random.choice(
        num_features, num_significant_features, replace=False
    )

    for j in range(num_features):
        mean_case = np.random.uniform(50, 70)
        std_case = np.random.uniform(1, 80)
        mean_control = np.random.uniform(50, 70)
        std_control = np.random.uniform(1, 80)

        case_data[:, j] = np.random.normal(mean_case, std_case, num_cases)
        control_data[:, j] = np.random.normal(mean_control, std_control, num_controls)

        # Add extra signal for significant features
        if j in significant_feature_indices:
            case_data[:, j] += np.random.normal(60, 60, num_cases)

    # Create DataFrames
    case_df = pd.DataFrame(case_data, columns=[f"feature_{i}" for i in range(num_features)])
    control_df = pd.DataFrame(control_data, columns=[f"feature_{i}" for i in range(num_features)])

    case_df["cancer"] = 1
    control_df["cancer"] = 0

    simulated_data = pd.concat([case_df, control_df], ignore_index=True)
    return case_df, control_df, simulated_data
