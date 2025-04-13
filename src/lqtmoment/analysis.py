"""
Data analysis module for the lqt-moment-magnitude package.

Version: 0.1.1

This module provides robust data analysis tools for lqt-moment-magnitude package. It uses
lqtmoment-formatted catalog data as input, constructs a class object from the data, and perform comprehensive
data analysis. Beyond statistical analysis capabilities, it also offers data visualization facilitated insights 
and interpretation.  

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import plotly.express as px
from typing import Optional, List, Callable, Tuple, Dict
from scipy.stats import linregress

from .utils import load_data



class LqtAnalysis:
    """
    A class for analyzing and visualizing lqtmoment catalog data.

    Attributes:
        data (pd.DataFrame): The lqtmoment-formatted catalog data stored as pandas
                            DataFrame.   

    Example:
        >>> from lqtmoment.analysis import LqtAnalysis, load_catalog
        >>> lqt_data = load_catalog(r"tests/data/catalog/")
        >>> mw_average = lqt_data.average('magnitude')
        >>> lqt_data.plot_histogram('magnitude') 
    """
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        """
        Initialized a lqtmoment data analysis object.

        Args:
            dataframe (Optional[pd.DataFrame]): A DataFrame object of full lqtmoment formatted catalog.
                                                Defaults to None and create empty class object.
        """
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")
            if dataframe.empty:
                raise ValueError("DataFrame cannot be empty")
        self.data = dataframe


    def _clean_column(self, column_name: str) -> pd.DataFrame:
        """ Helper function to clean and make sure the column is numeric."""
        if column_name not in self.data.columns:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame")
        df_column = self.data[column_name]
        if np.issubdtype(df_column.dtype, np.number):
            return df_column
        if df_column.empty:
            raise ValueError(f"Column {column_name} is empty")
        numeric_column = pd.to_numeric(df_column, errors='coerce')
        if numeric_column.isna().all():
            raise ValueError(f"Column {column_name} contains no valid numeric data")
        return numeric_column
    

    def average(self, column_name: str) -> float:
        """
        Calculate the average value of the specified column.   

        Args:
            column_name (str): Name of the column ot compute the mean for.
        
        Returns:
            float: The mean of the column if it contains valid numeric data.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If not DataFrame is provided.
        """
        if self.data is None:
            raise ValueError("No dataframe provided")
        return self._clean_column(column_name).mean()


    def median(self, column_name: str) -> float:
        """
        Calculate the median value of the specified column.   

        Args:
            column_name (str): Name of the column ot compute the median for
        
        Returns:
            float: The median of the column if it contains valid numeric data.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If not DataFrame is provided.
        """
        if self.data is None:
            raise ValueError("No dataframe provided")
        return self._clean_column(column_name).median()


    def std_dev(self, column_name: str) -> float:
        """
        Calculate the standard deviation value of the specified column.   

        Args:
            column_name (str): Name of the column ot compute the standard deviation for.
        
        Returns:
            float: The standard deviation of the column if it contains valid numeric data.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If not DataFrame is provided.
        """
        if self.data is None:
            raise ValueError("No dataframe provided")
        return self._clean_column(column_name).std()


    def minimum(self, column_name: str) -> float:
        """
        Calculate the minimum value of the specified column.
        
        Args:
            column_name (str): Name of the column to compute the minimum for.
        
        Returns:
            float: The minimum of the column if it contains valid numeric data.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, or it contains no valid numeric data.
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")
        return self._clean_column(column_name).min()
    

    def maximum(self, column_name: str) -> float:
        """
        Calculate the maximum value of the specified column.
        
        Args:
            column_name (str): Name of the column to compute the maximum for.
        
        Returns:
            float: The maximum of the column if it contains valid numeric data.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, or it contains no valid numeric data.
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")
        return self._clean_column(column_name).max()
    

    def describe(self, column_name) -> float:
        """
        Compute summary statistics for the specified column.
        
        Args:
            column_name (str): Name of the column to summarize.
        
        Returns:
            pd.Series: A Series containing count, mean, std, min, max, and quartiles.
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, or it contains no valid numeric data.
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")
        return self._clean_column(column_name).describe()
    
    
    def plot_histogram(
        self,
        column_name: str,
        bins: Optional[int] =10,
        save_figure: Optional[bool] = False
        ) -> None:
        """
        Plot histogram for the specific column.

        Args:
            column_name (str): Name of the column to plot the histogram for.
            bins (Optional[int]): The number of bins.
            save_figure (Optional[bool]): If true, save the plot. Defaults to False.
        
        Return:
            None
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, or it contains no valid numeric data.        
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")
        data = self._clean_column(column_name)
        plt.hist(data.dor)     
        
        
        return None
        




    def gutenberg_richter(
        self,
        column_name: Optional[str] = 'magnitude',
        min_magnitude: Optional[float] = None,
        bin_width: float = 0.1,
        plot: Optional[bool] = True,
        save_plot: Optional[bool] = False
        ) -> Dict:
        """
        Compute Gutenberg-Richter magnitude-frequency analysis and estimate the b-value.

        Args:
            column_name (Optional[str]): Name of the magnitude column. Defaults to 'magnitude'.
            min_magnitude (Optional[float]): Minimum magnitude threshold. If None, uses the
                                            minimum in the catalog.
            bin_width (Optional[float]): Width of magnitudes bins (e.g., 0.1 for 0.1-unit bins).
                                            Default is True.
            plot(Optional[bool]): If True, display a plot of the Gutenberg-Richter relationship. 
                                    Defaults is True.
        
        Returns:
            dict object contains:
                - 'b_value': Estimated b-value (slope of the linear fit).
                - 'a_value': Estimated a-value (intercept of the linear fit).
                - 'data': DataFrame with 'magnitude' and 'log_cumulative_count'
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, contains no valid numeric data,
                    or insufficient data for fitting.        
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")

        if bin_width <= 0:
            raise ValueError("bind_width must be positive")
        
        magnitudes = self._clean_column(column_name)
        valid_magnitudes = magnitudes.dropna()

        if len(valid_magnitudes) < 10:
            raise  ValueError("Insufficient valid data for Gutenberg-Richter analysis")
        
        if min_magnitude is None:
            min_magnitude = np.floor(valid_magnitudes.min() / bin_width) *bin_width
        elif min_magnitude > valid_magnitudes.max():
            raise ValueError("min magnitude exceeds maximum observed magnitude")
        
        filtered_magnitudes = valid_magnitudes[valid_magnitudes >= min_magnitude]
        if len(filtered_magnitudes) < 10:
            raise ValueError("Insufficient data above min_magnitude for analysis")
        
        max_magnitude = np.ceil(filtered_magnitudes.max() / bin_width) * bin_width
        mag_bins = np.arange(min_magnitude, max_magnitude + bin_width, bin_width)
        
        # Compute the cumulative counts
        cumulative_counts = [len(filtered_magnitudes[filtered_magnitudes >= m]) for m in mag_bins]

        # Compute non cumulative counts
        non_cumulative_counts = np.histogram(filtered_magnitudes, bins=mag_bins)[0]

        # Shift the non-cumulative bins to represent bin centers
        mag_bins_non_cum = mag_bins[:-1] + bin_width/2

        # Filter out zero counts to avoid log issues
        valid_count_indices_cum = [i for i, c in enumerate(cumulative_counts) if c > 0]
        valid_count_indices_non_cum = [i for i, c in enumerate(non_cumulative_counts) if c > 0]

        if len(valid_count_indices_cum) < 5:
            raise ValueError("Too few non-zero cumulative counts for reliable fitting")
        
        # Cumulative fit data
        mag_bins_cum = mag_bins[valid_count_indices_cum]
        cumulative_counts = cumulative_counts[valid_count_indices_cum]
        log_cumulative_counts = np.log10(cumulative_counts)

        # Non-cumulative data
        mag_bins_non_cum = mag_bins_non_cum[valid_count_indices_non_cum]
        non_cumulative_counts = non_cumulative_counts[valid_count_indices_non_cum]
        log_non_cumulative_counts = np.log10(non_cumulative_counts + 1e-10)


        # Linear fitting for b-value
        slope, intercept, r_value, _, stderr = linregress(mag_bins_cum, log_cumulative_counts)
        b_value = -slope
        a_value = intercept

        # Result

        result_data = pd.DataFrame(
            {
                'magnitude': mag_bins[:-1] + bin_width/2,
                'log_cumulative_count': np.full(len(mag_bins) - 1, np.nan),
                'log_non_cumulative_count': np.full(len(mag_bins) - 1, np.nan)
            }
        )

        cum_indices_map = [i for i, idx in enumerate(valid_count_indices_cum) if idx < len(mag_bins) - 1]
        non_cum_indices_map = [i for i, idx in enumerate(valid_count_indices_non_cum) if idx < len(mag_bins) - 1]
        result_data.loc[cum_indices_map, 'log_cumulative_count'] = log_cumulative_counts[cum_indices_map]
        result_data.loc[non_cum_indices_map, 'log_non_cumulative_count'] = log_non_cumulative_counts[non_cum_indices_map]

        result = {
            'b_value': b_value, 
            'a_value': a_value,
            'b_value_stderr': stderr, 
            'r_squared': r_value**2,
            'data': result_data
        }

        # Plotting
        if plot:
            plt.figure(figsize = (8,6))
            plt.scatter(mag_bins_cum, log_cumulative_counts, label = "Cumulative (N>=M)", color='blue', marker='o')
            plt.scatter(mag_bins_non_cum, log_non_cumulative_counts, label='Non-Cumulative (per bin)', color='green', marker='^')
            plt.plot(mag_bins_cum, slope * mag_bins_cum + intercept, color='red',
                     label=f"Fit b={b_value:2f}, R_square = {r_value**2:.2f}")
            plt.xlabel('Magnitude')
            plt.ylabel('Log10(Count)')
            plt.title(f"Gutenberg-Richter Analysis (bin_width = {bin_width})")
            plt.legend()
            if save_plot:
                plt.savefig("b_value_analysis.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        return result



        







def load_catalog(catalog_file: str) -> LqtAnalysis:
    """
    Load lqtmoment formatted catalog, this functions will handle
    data suffix/format (.xlsx or .csv) for more dynamic inputs

    Args:
        catalog_file (str): directory of the catalog file (e.g., .xlsx, .csv).
    
    Returns:
        LqtAnalysis: An initialized LqtAnalysis instance for data analysis.

    Raises: 
        FileNotFoundError: If the catalog file does not exist or cannot be read. 
        TypeError: If 
        ValueError: If the file format is unsupported 
    """
    try:
        dataframe = load_data(catalog_file)
        if dataframe.empty:
            raise ValueError(f"Catalog file '{catalog_file}' is empty")
        return LqtAnalysis(dataframe)
    except (FileNotFoundError, ValueError) as e:
        raise type(e)(f"Failed to load '{catalog_file}': {str(e)} ") from e
