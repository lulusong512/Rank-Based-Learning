import numpy as np
import pandas as pd


def modify_value(value, percent):
    """
    Randomly replaces a value with NaN based on a given probability.

    Parameters
    ----------
    value : float or int
        Original numerical value.
    percent : float
        Probability (between 0 and 1) of converting the value to NaN.

    Returns
    -------
    float or np.nan
        Returns np.nan with probability = `percent`, otherwise returns the original value.

    Notes
    -----
    This helper function is used to simulate missingness in the dataset.
    Example:
        modify_value(5.0, percent=0.1)  # 10% chance the output will be np.nan
    """
    random_num = np.random.random()  # Draw a random number between 0 and 1
    if random_num < percent:
        return np.nan  # Introduce missing value with given probability
    return value

# %%
def simulation_missing(random_seed: int, num_significant_features: int):
    """
    Generate simulated case-control data and progressively introduce missing values.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    num_significant_features : int
        Number of features significantly associated with the outcome, 
        passed to the `simulation_data()` generator.

    Returns
    -------
    df : pd.DataFrame
        Complete dataset without missing values.
    df_na : pd.DataFrame
        Dataset with ~10% missing values.
    df_na1 : pd.DataFrame
        Dataset with ~12% missing values.
    df_na2 : pd.DataFrame
        Dataset with additional ~12% missing values.
    df_na3 : pd.DataFrame
        Dataset with additional ~14% missing values.
    df_na4 : pd.DataFrame
        Dataset with additional ~17% missing values.

    Notes
    -----
    - This function uses `simulation_data()` to create the base dataset (n=60, p=300).
    - Missing values are introduced gradually across five stages using 
      `modify_value()` with increasing probabilities.
    - The 'cancer' outcome column (1=case, 0=control) is re-assigned after 
      missing values are introduced to preserve label integrity.
    """
    # Import simulation_data() from your existing module
    from .simulation_true_differential_60samples import simulation_data_true_differeial_60_samples as simulation_data

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Generate the base dataset (complete data)
    _, _, df = simulation_data(random_seed, num_significant_features)

    # Replace negative feature values (if any) with 0
    df[df < 0] = 0

    # -------------------------------------------------------
    # Generate progressive missingness versions of the dataset
    # -------------------------------------------------------
    # Each stage introduces additional missing values on top of the previous one
    df_na = df.apply(lambda x: x.apply(modify_value, percent=0.10))
    df_na1 = df_na.apply(lambda x: x.apply(modify_value, percent=0.12))
    df_na2 = df_na1.apply(lambda x: x.apply(modify_value, percent=0.12))
    df_na3 = df_na2.apply(lambda x: x.apply(modify_value, percent=0.14))
    df_na4 = df_na3.apply(lambda x: x.apply(modify_value, percent=0.17))

    # -------------------------------------------------------
    # Recreate binary outcome labels (1 = case, 0 = control)
    # -------------------------------------------------------
    # The first 30 rows correspond to case samples, the remaining 30 to controls
    for d in [df, df_na, df_na1, df_na2, df_na3, df_na4]:
        d["cancer"] = 0
        d.iloc[:30, -1] = 1

    return df, df_na, df_na1, df_na2, df_na3, df_na4
