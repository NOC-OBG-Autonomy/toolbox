import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd


def plot_time_series(
    data, x_var, y_var, title="Time Series Plot", xlabel=None, ylabel=None, **kwargs
):
    """Generates a time series plot for xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variables exist in the xarray dataset
        if x_var not in data.coords or y_var not in data:
            raise ValueError(
                f"Variables {x_var} and {y_var} must exist in the dataset."
            )
        x_data = data[x_var].values  # Extract x_data (usually time dimension)
        y_data = data[y_var].values  # Extract the y_data (variable to plot)
    elif isinstance(data, pd.DataFrame):
        # Fallback to Pandas DataFrame handling
        x_data = data[x_var]
        y_data = data[y_var]
    else:
        # Assuming custom format such as lists or arrays
        x_data, y_data = data[0], data[1]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, **kwargs)
    plt.xlabel(xlabel or x_var)
    plt.ylabel(ylabel or y_var)
    plt.title(title)
    plt.show()


def plot_histogram(data, var, bins=30, title="Histogram", xlabel=None, **kwargs):
    """Generates a histogram for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    elif isinstance(data, pd.DataFrame):
        # Fallback to Pandas DataFrame handling
        data_to_plot = data[var]
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins=bins, alpha=0.7, **kwargs)
    plt.xlabel(xlabel or var)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def plot_boxplot(data, var, title="Box Plot", xlabel=None, **kwargs):
    """Generates a box plot for a given variable in xarray data."""
    if isinstance(data, xr.Dataset):
        # Ensure that the variable exists in the xarray dataset
        if var not in data:
            raise ValueError(f"Variable {var} must exist in the dataset.")
        data_to_plot = data[var].values
    elif isinstance(data, pd.DataFrame):
        # Fallback to Pandas DataFrame handling
        data_to_plot = data[var]
    else:
        # Handle custom data types like lists or arrays
        data_to_plot = data

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel or var)
    plt.show()


def plot_correlation_matrix(data, variables=None, title="Correlation Matrix", **kwargs):
    """Generates a heatmap of the correlation matrix for xarray data."""
    if isinstance(data, xr.Dataset):
        if variables is None:
            variables = list(data.data_vars)  # Use all variables by default
        # Extract the variables to calculate the correlation matrix
        corr = data[variables].to_array().T.corr(dim="dim_0")
    elif isinstance(data, pd.DataFrame):
        corr = data.corr()
    else:
        raise TypeError(
            "Data must be a pandas DataFrame or xarray Dataset to generate correlation matrix."
        )

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, **kwargs)
    plt.title(title)
    plt.show()


def generate_summary_stats(data, **kwargs):
    """Generate summary statistics for a given dataset (works for both xarray and pandas)."""
    if isinstance(data, xr.Dataset):
        # For xarray, we'll summarize each data variable
        print("Summary Statistics:")
        for var in data.data_vars:
            print(f"{var}:")
            print(
                data[var].mean().values, data[var].std().values
            )  # Just showing mean and std for simplicity
            print()
    elif isinstance(data, pd.DataFrame):
        summary = data.describe().transpose()
        print("Summary Statistics:\n", summary)
    else:
        print(
            "Summary statistics only supported for xarray Dataset or pandas DataFrame."
        )


def check_missing_values(data):
    """Check for missing values in the dataset (works for both xarray and pandas)."""
    if isinstance(data, xr.Dataset):
        missing = data.isnull().sum()
        print("Missing Values in Xarray Dataset:\n", missing)
    elif isinstance(data, pd.DataFrame):
        missing = data.isnull().sum()
        print("Missing Values in Pandas DataFrame:\n", missing)
    else:
        print(
            "Missing value check only supported for xarray Dataset or pandas DataFrame."
        )
