# Import required libraries
import numpy as np
import pandas as pd


def simulation_data_true_differeial_200_samples(random_seed: int, num_significant_features: int):
    """
    Generate simulated case-control data with significant and non-significant features
    under a true differential scenario (200 total samples: 100 cases, 100 controls).

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    num_significant_features : int
        Number of features significantly associated with the outcome.

    Returns
    -------
    case_df : pd.DataFrame
        Simulated data for the case group.
    control_df : pd.DataFrame
        Simulated data for the control group.
    simulated_data : pd.DataFrame
        Combined dataset with a 'cancer' outcome column (1 = case, 0 = control).

    """

    # Set seed for reproducibility
    np.random.seed(random_seed)

    # Define dataset dimensions
    num_cases = 100
    num_controls = 100
    num_features = 300

    # Initialize arrays for simulated data
    case_data = np.zeros((num_cases, num_features))
    control_data = np.zeros((num_controls, num_features))

    # Randomly select indices of significant features
    significant_feature_indices = np.random.choice(
        num_features, num_significant_features, replace=False
    )

    # Generate data feature by feature
    for j in range(num_features):
        # Randomly draw mean and standard deviation for each feature
        mean_case = np.random.uniform(50, 70)
        std_case = np.random.uniform(1, 80)
        mean_control = np.random.uniform(50, 70)
        std_control = np.random.uniform(1, 80)

        # Generate feature values for cases and controls
        case_data[:, j] = np.random.normal(loc=mean_case, scale=std_case, size=num_cases)
        control_data[:, j] = np.random.normal(loc=mean_control, scale=std_control, size=num_controls)

        # Add extra signal to case group for significant features
        if j in significant_feature_indices:
            case_data[:, j] += np.random.normal(loc=60, scale=60, size=num_cases)

    # If there are very few significant features, add global noise
    if num_significant_features < 9:
        noise = np.random.normal(loc=25, scale=14, size=case_data.shape)
        case_data += noise
        control_data += noise

    # Convert arrays to DataFrames
    case_df = pd.DataFrame(case_data, columns=[f"feature_{i}" for i in range(num_features)])
    control_df = pd.DataFrame(control_data, columns=[f"feature_{i}" for i in range(num_features)])

    # Add outcome labels
    case_df["cancer"] = 1
    control_df["cancer"] = 0

    # Combine case and control datasets
    simulated_data = pd.concat([case_df, control_df], ignore_index=True)

    return case_df, control_df, simulated_data
