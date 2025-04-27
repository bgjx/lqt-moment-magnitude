"""
Data analysis module for the lqt-moment-magnitude package.

Version: 0.2.0

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
from typing import Optional, Dict
from scipy.stats import linregress
from enum import Enum
from datetime import datetime


from .utils import load_data

class Statistic(Enum):
    "Enumeration for statistical operations."
    MEAN = "mean"
    MEDIAN = "median"
    STD = 'std'
    MIN = 'min'
    MAX = 'max'
    DESCRIBE = 'describe'


class LqtAnalysis:
    """
    A class for analyzing and visualizing lqtmoment catalog data.

    Attributes:
        data (pd.DataFrame): The lqtmoment-formatted catalog data stored as pandas
                            DataFrame.   

    Examples:
    ``` python
        >>> from lqtmoment.analysis import LqtAnalysis, load_catalog
        >>> lqt_data = load_catalog(r"tests/data/catalog/")
        >>> mw_average = lqt_data.average('magnitude')
        >>> lqt_data.plot_histogram('magnitude')
    ``` 
    """
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        """
        Initialized a lqtmoment data analysis object.

        Args:
            dataframe (Optional[pd.DataFrame]): A DataFrame object of full lqtmoment formatted catalog.
                                                Defaults to None and create empty class object.
        """
        self.data = None
        self._cache_cleaned_column = {} 
        if dataframe is not None:
            self._set_dataframe(dataframe)
    

    def _set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """ Helper function to set the dataframe for analysis """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty")
        if 'source_id' not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'source_id' column")
        self.data = dataframe.copy()
        self._cache_cleaned_column.clear()


    def _clean_column(self, column_name: str) -> pd.Series:
        """ Helper function to clean and make sure the column is numeric."""
        # Check this column in cache first
        if column_name in self._cache_cleaned_column:
            return self._cache_cleaned_column[column_name]
        
        if self.data is None:
            raise ValueError("No DataFrame provided")
        if column_name not in self.data.columns:
            raise KeyError(f"Column {column_name} does not exist in the DataFrame")
        
        column_series = self.data[['source_id', column_name]].drop_duplicates(subset='source_id')[column_name] 

        if not np.issubdtype(column_series.dtype, np.number):
            column_series = pd.to_numeric(column_series, errors='coerce')

        if column_series.isna().all():
            raise ValueError(f"Column {column_name} contains no valid numeric data")
        self._cache_cleaned_column[column_name] = column_series
        return column_series
    

    def _validate_geo_columns(
        self,
        lat: pd.Series,
        lon: pd.Series
        ) -> None:
        """ Helper Function to validate geographic coordinate columns"""
        if not lat.between(-90, 90).all():
            raise ValueError("Latitude values must be between -90 and 90")
        if not lon.between(-180, 180).all():
            raise ValueError("Longitude must be between -180 and 180")
    

    def _render_figure(
        self,
        fig: go.Figure,
        filename: str,
        save_figure: bool = False
        )-> None:
        """ Helper function for rendering and saving the figure """
        if save_figure:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.write_image(f"{filename}_{timestamp}.png")
        else:
            fig.show()
    

    def head(self, row_number: int = 5):
        """
        Return the first ten rows of the input dataframe.

        Args:
            row_number (int): The number of how many rows should be display. Default to 5.
        Returns:
            First ten rows of the input dataframe.

        Examples:
        ``` python
            >>> df = pd.DataFrame({"magnitude": [1, 2, 3]})
            >>> lqt = LqtAnalysis(df)
            >>> lqt.head()
        ```
        """
        return self.data.head(row_number)


    def compute_statistic(self, column_name: str, statistic_op : Statistic) -> float:
        """
        Compute statistic operation for the specified column.

        Args:
            column_name(str): Name of the column.
            statistic_op (Statistic): Statistic operation to compute (mean, median, std, max, min)
        
        Returns:
            float: The result of the statistic operation.
        
        Raises:
            KeyError: If column_name does not exist.
            ValueError: If data is invalid.
        
        Examples:
        ``` python
            >>> df = pd.DataFrame({"magnitude": [1, 2, 3]})
            >>> lqt = LqtAnalysis(df)
            >>> lqt.compute_statistic("magnitude", Statistic.MEAN)
            2.0
        ```
        """
        data = self._clean_column(column_name)
        statistic_function = {
            Statistic.MEAN: data.mean,
            Statistic.MEDIAN: data.median,
            Statistic.STD: data.std,
            Statistic.MAX: data.max,
            Statistic.MIN: data.min,
            Statistic.DESCRIBE: data.describe
        }[statistic_op]
        return statistic_function()
       
    
    def window_time(
        self,
        min_time : str,
        max_time : str,
        ) -> pd.DataFrame:
        """
        Subset DataFrame by specific date time range.

        Args:
            min_time (str): A string following this format '%Y-%m-%d %H:%M:%S' as min time range.
            max_time (str): A string following this format '%Y-%m-%d %H:%M:%S' as min time range.
        
        Returns:
            pd.DataFrame: A subset from main DataFrame after time windowing.

        Raises:
            ValueError: Unmatched string format input.
        
        Examples:
        ``` python
            >>> df = pd.DataFrame({
            ...     "lat": [34.0, 34.1], "lon": [-118.0, -118.1],
            ...     "depth": [10, 12], "magnitude": [3.0, 3.5]
            ... })
            >>> lqt = LqtAnalysis(df)
            >>> subset_df = lqt.window_time(
            ...             min_tame = '2025-09-22 10:10:01',
            ...             max_time = '2025-09-25 10:10:01'
            ... )
        ```
        """
        try:
            min_datetime = datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
            max_datetime = datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise ValueError("Given time range follows unmatched format") from e
        
        if min_datetime > max_datetime:
            raise ValueError ("min_datetime must be earlier than max_datetime")
        if min_datetime < self.data['source_origin_time'].min() or max_datetime > self.data['source_origin_time'].max():
            raise ValueError("Given time ranges are outside catalog time range.")

        subset_df = self.data[(self.data['source_origin_time'] >= min_datetime ) & (self.data['source_origin_time'] <= max_datetime)]

        return subset_df
        
    
    def area_rectangle(
        self,
        min_latitude: float,
        max_latitude: float,
        min_longitude: float,
        max_longitude: float,       
        ) -> pd.DataFrame:
        """
        Subset DataFrame by specifying rectangle area.

        Args:
            min_latitude (float): Minimum latitude or North border.
            max_latitude (float): Maximum latitude or South border.
            min_longitude (float): Minimum longitude or West border.
            max_longitude (float): Maximum longitude or East border.
        
        Returns:
            pd.DataFrame: A subset from main DataFrame based on rectangle area.
        
        Raises:
            ValueError: Rectangle area outside the main DataFrame.

        Examples:
        ``` python
            >>> df = pd.DataFrame({
            ...     "lat": [34.0, 34.1], "lon": [-118.0, -118.1],
            ...     "depth": [10, 12], "magnitude": [3.0, 3.5]
            ... })
            >>> lqt = LqtAnalysis(df)
            >>> subset_df = lqt.window_time(
            ...             min_latitude = 34.0,
            ...             max_latitude = 34.1,
            ...             min_longitude = -118.1,
            ...             max_longitude = -118.0,
            ... )
        ```     
        """
        self._validate_geo_columns(min_latitude, min_longitude)
        self._validate_geo_columns(max_latitude, max_longitude)

        # Check if are rectangle outside the DataFrame
        if min_latitude > max_latitude: 
            raise ValueError("min_latitude must be smaller than max_latitude")
        if min_longitude > max_longitude:
            raise ValueError("min_longitude must be smaller than max_longitude")
        
        if min_latitude < self.data['source_lat'].min() or max_latitude > self.data['source_lat'].max():
            raise ValueError("Given latitude range are outside catalog area coverage")
        if min_longitude < self.data['source_lon'].min() or max_longitude > self.data['source_lon'].max():
            raise ValueError("Given longitude range are outside catalog area coverage")

        subset_df = self.data[
            (self.data['latitude'] >= min_latitude) & 
            (self.data['latitude'] <= max_latitude) & 
            (self.data['longitude'] >= min_longitude) & 
            (self.data['longitude'] <= max_longitude)
            ]

        return subset_df
    

    def plot_histogram(
        self,
        column_name: str,
        bin_width: Optional[float] = None,
        min_bin: Optional[float] = None,
        max_bin: Optional[float] = None,
        save_figure: bool = False
        ) -> None:
        """
        Plot a histogram for the specific column with manual binning.

        Args:
            column_name (str): Name of the column to plot the histogram for.
            bin_width (Optional[float]): Determine the bin width. Defaults to None,
                                        trigger automatic binning.
            min_bin (float): Minimum bin edge. Default to None, min bin will be calculated
                                automatically.
            max_bin (float): Maximum bin edge. Default to None, max bin will be calculated
                                automatically.
            save_figure (bool): If true, save the plot. Defaults to False.
        
        Returns:
            None
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, or it contains no valid numeric data.
            TypeError: If min_edge, max_edge, or bin_width are not numeric.

        Examples:
        ``` python
            >>> df = pd.DataFrame({"magnitude": [1, 2, 2, 3]})
            >>> lqt = LqtAnalysis(df)
            >>> lqt.plot_histogram("magnitude", bin_width=0.5)
        ```       
        """

        # clean and validate the column data
        data = self._clean_column(column_name).dropna()
        if data.empty:
            raise ValueError(f"No valid data available for plotting in column {column_name}")
        
        # Validate the min_bin and max_bin
        if min_bin is not None and not isinstance(min_bin, (int, float)):
            raise TypeError("min_bin must be a numeric value")
        if max_bin is not None and not isinstance(max_bin, (int, float)):
            raise TypeError("max_bin must be a numeric value")
        if min_bin is not None and max_bin is not None and min_bin >= max_bin:
            raise ValueError("min_bin must be greater than the max_bin")
        

        # Compute bins
        if bin_width is None:
            raise ValueError("bin_width must be provided for manual binning")
        if not isinstance(bin_width, (int, float)) or bin_width <= 0:
            raise ValueError("bin_width must be a positive numeric value")
            
        # Use user provided min_bin and max_bin, otherwise fall back to data min/max
        min_val = min_bin if min_bin is not None else np.floor(data.min() / bin_width) * bin_width
        max_val = max_bin if max_bin is not None else np.ceil(data.max() / bin_width) * bin_width

        # Create bin edges
        bin_edges = np.arange(min_val, max_val + bin_width, bin_width)


        # Calculate bin centers (midpoints of bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Populate data into bins (count occurrences in each bin)
        hist_counts, _ = np.histogram(data, bins = bin_edges)

        # Create plotly figure
        fig = go.Figure()

        # Add histogram bars using bin centers and counts
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist_counts,
                width=bin_width,
                name=column_name,
                hovertemplate="Bin Center: %{x:.3f}<br>Count: %{y}<extra></extra>",
                marker= dict(color = 'skyblue')
            )
        )

        # Figure layout
        fig.update_layout(
            title = f"Histogram of {column_name}",
            xaxis_title = column_name,
            yaxis_title = "Count",
            showlegend = False,
            bargap = 0.2,
            template='plolty_white',
            xaxis=dict(
                tickmode = 'array',
                tickvals=bin_centers,
                ticktext=[f"{x:.3f}" for x in bin_centers]
            )
        )
        
        self._render_figure(fig, f"histogram_{column_name}", save_figure)

        return None
    

    def plot_hypocenter_3d(
        self,
        lat_column: str,
        lon_column: str,
        depth_column: str,
        color_by: Optional[str] = None ,
        size_by: Optional[str] = None,
        save_figure: bool = False
        ) -> None:
        """
        Create interactive 2D or 3D hypocenter plot.

        Args:
            lat_column (str): Name of the latitude column.
            lon_column (str): Nme of longitude column.
            depth_column (str) : Name of the depth column.
            color_by (Optional[str]): Name of the column to map color points by. If None,
                                        use single color.
            size_by (Optional[str]): Name of the column to map size points by. If None,
                                        use default size.
            save_figure (bool): If True, save the plot.
        
        Raises:
            KeyError: If any specified column does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, columns are empty, or contain no valid numeric data.

        Examples:
        ``` python
            >>> df = pd.DataFrame({
            ...     "lat": [34.0, 34.1], "lon": [-118.0, -118.1],
            ...     "depth": [10, 12], "magnitude": [3.0, 3.5]
            ... })
            >>> lqt = LqtAnalysis(df)
            >>> lqt.plot_hypocenter_3d("lat", "lon", "depth", color_by="magnitude")
        ```    
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")

        # Get the data
        lat = self._clean_column(lat_column)
        lon = self._clean_column(lon_column)
        depth = self._clean_column(depth_column) if depth_column else None
        color = self._clean_column(color_by) if color_by else None
        size = self._clean_column(size_by) if size_by else None

        # Validate the geographic coordinate
        self._validate_geo_columns(lat, lon)

        # Combine the data to a Dict object and drop NaN
        data = pd.DataFrame(
            {
                'lat': lat,
                'lon': lon,
                'depth': depth,
                'color': color if color else np.ones_like(lat),
                'size': size if size else np.ones_like(lat)
            }
        ).dropna()

        if data.empty:
            raise ValueError("No valid data available for plotting after removing NaN values")
        
        if size_by:
            data['size'] = (data['size'] - data['size'].min()) / (data['size'].max() - data['size'].min() + 1e-10) * 10
        
        # Plotting the Data
        fig = px.scatter_3d(
            data,
            x='lon',
            y='lat',
            z='depth',
            color= 'color' if color_by else None,
            size = 'size' if size_by else None,
            color_continuous_scale= 'Varidis',
            hover_data = {'lat': ':.2f', 'lon': ':.2f', 'depth': ':.2f', 'color': ':.2f'},
            title= 'Earthquake Locations (3D)'
        )
        fig.update_scenes(
            xaxis_title = "Longitude",
            yaxis_title = "Latitude",
            zaxis_title = 'Depth (m)'
        )
        fig.update_layout(
            showlegend = bool(color_by),
            coloraxis_colorbar_title = color_by if color_by else None,
            template = 'plotly_white'
        )

        self._render_figure(fig, "3d_plot_hypocenter", save_figure)


    def plot_hypocenter_2d(
        self,
        lat_column: str,
        lon_column: str,
        color_by: Optional[str] = None ,
        size_by: Optional[str] = None,
        save_figure: bool = False
        ) -> None:
        """
        Create interactive 2D or 3D hypocenter plot.

        Args:
            lat_column (str): Name of the latitude column.
            lon_column (str): Nme of longitude column.
            color_by (Optional[str]): Name of the column to map color points by. If None,
                                        use single color.
            size_by (Optional[str]): Name of the column to map size points by. If None,
                                        use default size.
            save_figure (bool): If True, save the plot.
        
        Raises:
            KeyError: If any specified column does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, columns are empty, or contain no valid numeric data.

        Examples:
        ``` python
            >>> lqt.plot_hypocenter_2d("lat", "lon", color_by="magnitude")
        ```  
        """

        if self.data is None:
            raise ValueError("No DataFrame provided")

        # Get the data
        lat = self._clean_column(lat_column)
        lon = self._clean_column(lon_column)
        color = self._clean_column(color_by) if color_by else None
        size = self._clean_column(size_by) if size_by else None

        # Validate the geographic coordinate
        self._validate_geo_columns(lat, lon)

        # Combine the data to a Dict object and drop NaN
        data = pd.DataFrame(
            {
                'lat': lat,
                'lon': lon,
                'color': color if color else np.ones_like(lat),
                'size': size if size else np.ones_like(lat)
            }
        ).dropna()

        if data.empty:
            raise ValueError("No valid data available for plotting after removing NaN values")
        
        if size_by:
            data['size'] = (data['size'] - data['size'].min()) / (data['size'].max() - data['size'].min() + 1e-10) * 10
        
        # Plotting the Data
        fig = px.scatter(
            data,
            x='lon',
            y='lat',
            color= 'color' if color_by else None,
            size = 'size' if size_by else None,
            color_continuous_scale= 'Varidis',
            hover_data = {'lat': ':.2f', 'lon': ':.2f', 'depth': ':.2f', 'color': ':.2f'},
            title= 'Earthquake Locations (2D)'
        )

        fig.update_layout(
            showlegend = bool(color_by),
            coloraxis_colorbar_title = color_by if color_by else None,
            template = 'plotly_white',
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )

        self._render_figure(fig, "2d_plot_hypocenter", save_figure)


    def gutenberg_richter(
        self,
        column_name: str = 'magnitude',
        min_magnitude: Optional[float] = None,
        bin_width: float = 0.1,
        plot: bool = True,
        save_figure: bool = False
        ) -> Dict:
        """
        Compute Gutenberg-Richter magnitude-frequency analysis and estimate the b-value.

        Args:
            column_name (str): Name of the magnitude column. Defaults to 'magnitude'.
            min_magnitude (Optional[float]): Minimum magnitude threshold. If None, uses the
                                            minimum in the catalog.
            bin_width (float): Width of magnitudes bins (e.g., 0.1 for 0.1-unit bins).
                                            Default is True.
            plot(bool): If True, display a plot of the Gutenberg-Richter relationship. 
                                    Defaults is True.
            save_figure (bool): If true, save the plot. Defaults to False.
        
        Returns:
            dict object contains:
                - 'b_value': Estimated b-value (slope of the linear fit).
                - 'a_value': Estimated a-value (intercept of the linear fit).
                - 'b_value_stderr': Standard error of the b-value.
                - 'r_squared': R-squared value of the fit.
                - 'data': DataFrame with 'magnitude' and 'log_cumulative_count'
        
        Raises:
            KeyError: If the column_name does not exist in the DataFrame.
            ValueError: If no DataFrame is provided, the column is empty, contains no valid numeric data,
                    or insufficient data for fitting. 

        Examples:
        ``` python
            >>> df = pd.DataFrame({"magnitude": [3.0, 3.5, 4.0, 4.5, 5.0]})
            >>> lqt = LqtAnalysis(df)
            >>> result = lqt.gutenberg_richter(bin_width=0.5)
            >>> print(result['b_value'])
        ```
        """
        if bin_width <= 0:
            raise ValueError("bind_width must be positive")
        
        valid_magnitudes = self._clean_column(column_name).dropna()
        if len(valid_magnitudes) < 10:
            raise  ValueError("Insufficient valid data for Gutenberg-Richter analysis")
        
        data_range = valid_magnitudes.max() - valid_magnitudes.min()
        if bin_width > data_range / 2:
            raise ValueError(f"bin width {bin_width} is too large for data range ({data_range})")

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
        log_non_cumulative_counts = np.log10(non_cumulative_counts)

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
            fig = go.Figure()

            # Cumulative plot
            fig.add_trace(go.Scatter(
                x = mag_bins_cum,
                y = log_cumulative_counts,
                mode = 'markers',
                name = 'Cumulative (N >= M)',
                marker = dict(symbol = 'circle', color= 'blue',  size=8),
                hovertemplate = 'Magnitude: %{x:.3f}<br>Log10(Count): %{y:.2f}'
            ))

            # Non-cumulative scatter (triangles)
            fig.add_trace(go.Scatter(
                x = mag_bins_non_cum,
                y = log_non_cumulative_counts,
                mode= 'markers',
                name= 'Non-Cumulative (per bin)',
                marker = dict(symbol='triangle-up', color='green', size=8),
                hovertemplate = 'Magnitude: %{x:.3f}<br>Log10(Count): %{y:.2f}'

            ))

            # Fitted line
            fit_x = np.array([mag_bins_cum.min(), mag_bins_cum.max()])
            fit_y = slope * mag_bins_cum + intercept
            fig.add_trace(
                go.Scatter(
                    x = fit_x,
                    y = fit_y,
                    mode = 'Lines',
                    name = f"Fit: b={b_value:.2f}, R_square = {r_value**2:.2f}",
                    line = dict(
                        color='red'
                    )
                )
            )

            # Layout
            fig.update_layout(
                title = f"Gutenberg-Richter Analysis (bin_width = {bin_width})",
                xaxis_title = "Magnitude",
                yaxis_title = "Log10(Count)",
                legend = dict(x=0.7, y=0.9),
                showlegend = True,
                template = 'plotly_white'
            )

            self._render_figure(fig, "gutenberg_ritcher", save_figure)
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
