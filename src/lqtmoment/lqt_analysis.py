"""
Data analysis module for the lqt-moment-magnitude package.

Version: 0.2.1

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
    def __init__(self, dataframe: Optional[pd.DataFrame] = None) -> None:
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

        self._cache_cleaned_column[column_name] = column_series
        return column_series
    

    def _clean_column_numeric(self, column_name: str) -> pd.Series:
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
            fig.write_image(f"LqtAnalysis_figures/{filename}_{timestamp}.png")
        else:
            fig.show()
    

    def head(self, row_number: int = 5):
        """
        Return the first ten rows of the input dataframe.

        Args:
            row_number (int): The number of how many rows should be display. Default to 5.

        Returns:
            User specified first number of rows to be displayed.

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
        data = self._clean_column_numeric(column_name)
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
            min_time (str): A string following this format '%YYYY-%mm-%dd %HH:%MM:%SS' as min time range.
            max_time (str): A string following this format '%YYYY-%mm-%dd %HH:%MM:%SS' as min time range.
        
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
        
        # Convert the source origin time to datetime objcect
        source_origin_times = pd.to_datetime(self.data['source_origin_time'])
        if min_datetime > max_datetime:
            raise ValueError ("min_datetime must be earlier than max_datetime")
        if min_datetime < source_origin_times.min() or max_datetime > source_origin_times.max():
            raise ValueError("Given time ranges are outside catalog time range.")

        subset_df = self.data[(source_origin_times >= min_datetime ) & (source_origin_times <= max_datetime)]

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
        self._validate_geo_columns(pd.Series([min_latitude]), pd.Series([min_longitude]))
        self._validate_geo_columns(pd.Series([max_latitude]), pd.Series([max_longitude]))

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
            (self.data['source_lat'] >= min_latitude) & 
            (self.data['source_lat'] <= max_latitude) & 
            (self.data['source_lon'] >= min_longitude) & 
            (self.data['source_lon'] <= max_longitude)
            ]

        return subset_df
    

    def plot_histogram(
        self,
        column_name: str,
        bin_width: Optional[float] = None,
        min_bin: Optional[float] = None,
        max_bin: Optional[float] = None,
        plot_width: int = 960,
        plot_height: int = 720,
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
            plot_width (int): The width of the plot. Default to 960.
            plot_height (int): The height of the plot. Default to 720.
        
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
        data = self._clean_column_numeric(column_name).dropna()
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
        bin_edges = np.arange(min_val, (max_val + bin_width), bin_width)

        # Calculate bin centers
        bin_centers = (bin_edges[:-1]  + bin_edges[1:]) /2

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
                hovertemplate="Bin_center: %{x:.3f}<br>count: %{y}<extra></extra>",
                marker= dict(color = 'skyblue')
            )
        )

        # Figure layout
        fig.update_layout(
            width = plot_width,
            height = plot_height,
            title = f"Histogram of {column_name}",
            xaxis_title = column_name,
            yaxis_title = "Count",
            showlegend = True,
            bargap = 0.25,
            template='plotly_white',
            xaxis=dict(
                tickmode = 'array',
                tickvals=bin_centers,
                ticktext=[f"{x:.3f}" for x in bin_centers]
            ),
            legend=dict(
                yanchor = "top",
                y = 0.99,
                xanchor = "right",
                x = 0.99,             
                bgcolor="rgba(255,255,255,0.5)"
                
            )
        )
        
        fig.show()

        return None
    

    def plot_hypocenter_3d(
        self,
        plot_width: int = 960,
        plot_height: int = 720,
        ) -> None:
        """
        Create interactive 2D or 3D hypocenter plot.

        Args:
            plot_width (int): The width of the plot. Default to 960.
            plot_height (int): The height of the plot. Default to 720.
        
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
            >>> lqt.plot_hypocenter_3d()
        ```    
        """
        if self.data is None:
            raise ValueError("No DataFrame provided")

        # Get the Hypocenter Detailed Info
        source_id = self.data['source_id'].drop_duplicates()
        source_lat = self._clean_column_numeric('source_lat')
        source_lon = self._clean_column_numeric('source_lon')
        source_depth_m = self._clean_column_numeric('source_depth_m')
        source_magnitude = self._clean_column_numeric('magnitude') if  'magnitude' in self.data.columns else np.ones_like(source_lat)

        # Flipped the source depth
        source_depth_m = source_depth_m * -1

       # Get the Station Detailed Info
        sta_code = self._clean_column('station_code')
        sta_lat = self._clean_column_numeric('station_lat')
        sta_lon = self._clean_column_numeric('station_lon')
        sta_elev_m = self._clean_column_numeric('station_elev_m')

        # Validate the hypocenters geographic coordinate
        self._validate_geo_columns(source_lat, source_lon)

        # Validate the stations geographic coordinate
        self._validate_geo_columns(source_lat, source_lon)

        # Combine data and build hypocenters dataframe
        hypo_data = pd.DataFrame(
            {
                'source_id': source_id,
                'source_lat': source_lat,
                'source_lon': source_lon,
                'source_depth_m': source_depth_m,
                'magnitude': source_magnitude
            }
        ).dropna()

        # Combine data and build stations dataframe
        sta_data  = pd.DataFrame(
            {
                'sta_code': sta_code,
                'sta_lat': sta_lat,
                'sta_lon': sta_lon,
                'sta_elev_m': sta_elev_m
            }   
        )

        if hypo_data.empty:
            raise ValueError("No valid hypocenters data available for plotting after removing NaN values")
        
        if sta_data.empty:
            raise ValueError("No valid data stations available for plotting after removing NaN values")

        # Normalize the magnitude sizing
        if 'magnitude' in self.data.columns:
            hypo_data['normalized_magnitude'] = (hypo_data['magnitude'] - hypo_data['magnitude'].min()) / (hypo_data['magnitude'].max() - hypo_data['magnitude'].min() + 1e-10) * 1
                
        # Plotting the Data
        fig = px.scatter_3d(
            hypo_data,
            x='source_lon',
            y='source_lat',
            z='source_depth_m',
            color= 'source_depth_m',
            size = 'normalized_magnitude',
            color_continuous_scale= 'Viridis',
            custom_data=['source_id', 'source_depth_m', 'magnitude'],
            title= 'Earthquake Locations (3D)'
        )

        fig.update_traces(
            hovertemplate =
                "<b>Earthquake: %{customdata[0]}</b><br>" +
                "Source Lon: %{x:.3f}<br>" +
                "Source Lat: %{y:.3f}<br>" +
                "Source Elev (m): %{customdata[1]:.3f}<br>" +
                "Magnitude: %{customdata[2]:.3f}<br>" +
                "<extra></extra>",        
        )

        fig.add_trace(
            go.Scatter3d(
                x=sta_data['sta_lon'],
                y=sta_data['sta_lat'],
                z=sta_data['sta_elev_m'],
                mode='markers+text',
                marker = dict(
                    size=12,
                    symbol='circle',
                    color='red'
                ),
                name='Stations',
                text= sta_data['sta_code'],
                textposition='top center',
                textfont=dict(
                    size=12,
                    color='#333333'
                ),
                hovertemplate='<b>%{text}</b><br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>'
            )
        )

        fig.update_layout(
            width = plot_width,
            height = plot_height,
            showlegend = True,
            coloraxis_colorbar_title = 'elev_m',
            template = 'plotly_white',
            scene = dict(
                xaxis_title = "Longitude",
                yaxis_title = "Latitude",
                zaxis_title = 'Elev (m)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=0.5, y=0.5, z=0.5)
                )
            ),
            legend=dict(
                yanchor = "top",
                y = 0.99,
                xanchor = "left",
                x = 0.01,             
                bgcolor="rgba(255,255,255,0.5)"
                
            )
        )

        fig.show()

        return None


    def plot_hypocenter_2d(
        self,
        plot_width: int = 960,
        plot_height: int = 720,
        ) -> None:
        """
        Create interactive 2D or 3D hypocenter plot.

        Args:
            plot_width (int): The width of the plot. Default to 960.
            plot_height (int): The height of the plot. Default to 720.
       
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
            >>> lqt.plot_hypocenter_2d()
        ```  
        """

        if self.data is None:
            raise ValueError("No DataFrame provided")

        # Get the Hypocenter Detailed Info
        source_id = self.data['source_id'].drop_duplicates()
        source_lat = self._clean_column_numeric('source_lat')
        source_lon = self._clean_column_numeric('source_lon')
        source_depth_m = self._clean_column_numeric('source_depth_m')
        source_magnitude = self._clean_column_numeric('magnitude') if  'magnitude' in self.data.columns else np.ones_like(source_lat)

        # Get the Station Detailed Info
        sta_code = self._clean_column('station_code')
        sta_lat = self._clean_column_numeric('station_lat')
        sta_lon = self._clean_column_numeric('station_lon')

        # Validate the hypocenters geographic coordinate
        self._validate_geo_columns(source_lat, source_lon)

        # Validate the stations geographic coordinate
        self._validate_geo_columns(source_lat, source_lon)

        # Combine data and build hypocenters dataframe
        hypo_data = pd.DataFrame(
            {
                'source_id': source_id,
                'source_lat': source_lat,
                'source_lon': source_lon,
                'source_depth_m': source_depth_m,
                'magnitude': source_magnitude
            }
        ).dropna()

        # Combine data and build stations dataframe
        sta_data  = pd.DataFrame(
            {
                'sta_code': sta_code,
                'sta_lat': sta_lat,
                'sta_lon': sta_lon
            }   
        )

        if hypo_data.empty:
            raise ValueError("No valid hypocenters data available for plotting after removing NaN values")
        
        if sta_data.empty:
            raise ValueError("No valid data stations available for plotting after removing NaN values")

        # Normalize the magnitude sizing
        if 'magnitude' in self.data.columns:
            hypo_data['normalized_magnitude'] = (hypo_data['magnitude'] - hypo_data['magnitude'].min()) / (hypo_data['magnitude'].max() - hypo_data['magnitude'].min() + 1e-10) * 1
        
        # Find the center plot, and set zoom factor in (degrees)
        lat_center = source_lat.mean()
        lon_center = source_lon.mean()

        # Find the zoom span
        lat_span = abs(source_lat.max() - source_lat.min())
        lon_span = abs(source_lon.max() - source_lon.min())
        zoom_span = (lat_span + lon_span)/4

        # Plotting the Data
        fig = px.scatter(
            hypo_data,
            x='source_lon',
            y='source_lat',
            color= 'source_depth_m',
            size = 'normalized_magnitude',
            custom_data=['source_id', 'source_depth_m', 'magnitude'],
            color_continuous_scale= 'Viridis',
            title= 'Earthquake Locations (2D)'
        )

        fig.update_traces(
            hovertemplate =
                "<b>Earthquake: %{customdata[0]}</b><br>" +
                "Source Lon: %{x:.3f}<br>" +
                "Source Lat: %{y:.3f}<br>" +
                "Source Depth(m): %{customdata[1]:.3f}<br>" +
                "Magnitude: %{customdata[2]:.3f}<br>" +
                "<extra></extra>",        
        )

        fig.add_trace(
            go.Scatter(
                x=sta_data['sta_lon'],
                y=sta_data['sta_lat'],
                mode='markers+text',
                marker = dict(
                    size=16,
                    symbol='triangle-down',
                    color='red'
                ),
                name='Stations',
                text= sta_data['sta_code'],
                textposition='top center',
                textfont=dict(
                    size=12,
                    color='#333333'
                ),
                hovertemplate='<b>%{text}</b><br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>'
            )
        )

        fig.update_layout(
            width = plot_width,
            height = plot_height,
            showlegend = True,
            coloraxis_colorbar_title = 'depth_m',
            template = 'plotly_white',
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            xaxis = dict(
                range = [lon_center - zoom_span / 2, lon_center + zoom_span / 2],
                scaleanchor="y",
                scaleratio=1
            ),

            yaxis = dict(
                range = [lat_center - zoom_span / 2, lat_center + zoom_span / 2],
                scaleratio=1
            ),
            legend=dict(
                yanchor = "top",
                y = 0.99,
                xanchor = "left",
                x = 0.01,             
                bgcolor="rgba(255,255,255,0.5)"
                
            )
        )

        fig.show()

        return None


    def gutenberg_richter(
        self,
        min_magnitude: Optional[float] = None,
        bin_width: float = 0.1,
        plot: bool = True,
        plot_width: int = 960,
        plot_height: int = 720,
        ) -> Dict:
        """
        Compute Gutenberg-Richter magnitude-frequency analysis and estimate the b-value.

        Args:
            min_magnitude (Optional[float]): Minimum magnitude threshold. If None, uses the
                                            minimum in the catalog.
            bin_width (float): Width of magnitudes bins (e.g., 0.1 for 0.1-unit bins).
                                            Default is True.
            plot(bool): If True, display a plot of the Gutenberg-Richter relationship. 
                                    Defaults is True.
            plot_width (int): The width of the plot. Default to 960.
            plot_height (int): The height of the plot. Default to 720.
       
        
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
        
        valid_magnitudes = self._clean_column_numeric('magnitude').dropna()
        if len(valid_magnitudes) < 10:
            raise  ValueError("Insufficient valid data for Gutenberg-Richter analysis")
        
        data_range = valid_magnitudes.max() - valid_magnitudes.min()
        if bin_width > data_range / 2:
            raise ValueError(f"bin width {bin_width} is too large for data range ({data_range})")

        # set and check minimum magnitude integrity
        if min_magnitude is None:
            min_magnitude = np.floor(valid_magnitudes.min() / bin_width) *bin_width
        elif min_magnitude > valid_magnitudes.max():
            raise ValueError("min magnitude exceeds maximum observed magnitude")
        
        filtered_magnitudes = valid_magnitudes[valid_magnitudes >= min_magnitude]
        if len(filtered_magnitudes) < 10:
            raise ValueError("Insufficient data above min_magnitude for analysis")
        
        # Set and check maximum magnitude integrity
        max_magnitude = np.ceil(filtered_magnitudes.max() / bin_width) * bin_width

        # Set the magnitude bins
        mag_bins = np.arange(min_magnitude, max_magnitude + bin_width, bin_width)
        
        # Compute the cumulative counts
        cumulative_counts = np.array([len(filtered_magnitudes[filtered_magnitudes >= m]) for m in mag_bins])

        # Compute non cumulative counts
        non_cumulative_counts, _ = np.histogram(filtered_magnitudes, bins=mag_bins)

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

        
        # Fitting for determining b-value, a-value, and magnitude completeness
        # Parameters holder to find best parameters
        best_r_squared = 0
        best_breakpoint = mag_bins_cum[0]
        best_slope = 0
        best_intercept = 0
        best_index = 0
        best_std_err = 0

        # Search best break points (only use 50% of the data to speed up calculation)
        for i in range(1, len(mag_bins_cum)//2):
            breakpoint = mag_bins_cum[i]
            mask = mag_bins_cum >= breakpoint
            x_subset = mag_bins_cum[mask]
            y_subset = log_cumulative_counts[mask]

            if len(x_subset)< 2: 
                continue

            slope, intercept, r_value, _, std_err = linregress(x_subset, y_subset)
            r_squared = r_value**2

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_breakpoint = breakpoint
                best_slope = slope
                best_intercept = intercept
                best_index = i
                best_std_err = std_err

        # Fitted line
        fit_log_cumulative = (best_slope * mag_bins_cum) + best_intercept

        # Find the magnitude completeness, and set result dictionary
        mc = (best_breakpoint, fit_log_cumulative[best_index])
        b_value = -1 * best_slope
        a_value = best_intercept
        stderr = best_std_err
        r_value = best_r_squared
        
        result = {
            'b_value': b_value, 
            'a_value': a_value,
            'b_value_stderr': stderr, 
            'r_squared': r_value**2
        }

        # Plotting
        if plot:
            fig = go.Figure()

            # Cumulative plot scatter (dots)
            fig.add_trace(
                go.Scatter(
                    x = mag_bins_cum,
                    y = log_cumulative_counts,
                    mode = 'markers',
                    name = 'Cumulative (N >= M)',
                    marker = dict(symbol = 'circle', color= 'blue',  size=8),
                    hovertemplate = 'Magnitude: %{x:.3f}<br>Log10(Count): %{y:.2f}'
                )
            )

            # Non-cumulative scatter (triangles)
            fig.add_trace(
                go.Scatter(
                    x = mag_bins_non_cum,
                    y = log_non_cumulative_counts,
                    mode= 'markers',
                    name= 'Non-Cumulative (per bin)',
                    marker = dict(symbol='triangle-up', color='green', size=8),
                    hovertemplate = 'Magnitude: %{x:.3f}<br>Log10(Count): %{y:.2f}'
                )
            )

            # Plot the fitted line (lines)
            fig.add_trace(
                go.Scatter(
                    x = mag_bins_cum,
                    y = fit_log_cumulative,
                    mode = 'lines',
                    name = f"Fit: b_val={b_value:.2f}, a_val = {a_value:.2f}, R^2 = {r_value**2:.2f}",
                    line = dict(
                        color='red'
                    )
                )
            )

            # Plot the mc values (dot)
            fig.add_trace(
                go.Scatter(
                    x = [mc[0]],
                    y = [mc[1]],
                    mode='markers+text',
                    marker = dict(
                        size=12,
                        symbol='triangle-down',
                        color='red'
                    ),
                    name=f"Magnitude Completeness: {mc[0]:.2f}",
                    text= 'MC',
                    textposition='top center',
                    textfont=dict(
                        size=12,
                        color='#333333'
                    ),
                    hovertemplate='<b>Magnitude Completeness</b><br>Log10(N >= M): %{y:.2f}<br>Magnitude: %{x:.2f}<extra></extra>'
                    )
            )

            # Layout
            fig.update_layout(
                width = plot_width,
                height = plot_height,
                title = f"Gutenberg-Richter Analysis (bin_width = {bin_width})",
                xaxis_title = "Magnitude",
                yaxis_title = "Log10(Count)",
                showlegend = True,
                template = 'plotly_white',
                legend=dict(
                    yanchor = "top",
                    y = 0.99,
                    xanchor = "right",
                    x = 0.99,             
                    bgcolor="rgba(255,255,255,0.5)"
                )
            )

            fig.show()

        return result
    

    def plot_intensity(
        self,
        interval: str = 'monthly',
        plot_width: int = 960,
        plot_height: int = 720,
        ) -> None:
        """
        Calculate and plot histogram of earthquakes intensity in interval time defined by User.

        Args:
            interval (str): The time interval to calculate the earthquake intensity.
                            It should be 'yearly', 'monthly', 'weekly', 'daily', or 'hourly'.
            plot_width (int): The width of the plot. Default to 960.
            plot_height (int): The height of the plot. Default to 720.
        
        Raises:
            ValueError: If the given interval not in acceptable time interval.

        Returns: 
            None
        
        Examples:
        ``` python
            >>> df = pd.DataFrame({
            ...     "lat": [34.0, 34.1], "lon": [-118.0, -118.1],
            ...     "depth": [10, 12], "magnitude": [3.0, 3.5]
            ... })
            >>> lqt = LqtAnalysis(df)
            >>> lqt.plot_intensity()
        ```  
        """
        if interval.strip().lower() not in ['yearly', 'monthly', 'weekly', 'daily', 'hourly']:
            raise ValueError ("Your given interval is not valid, must be 'yearly', 'monthly', 'weekly', 'daily', or 'hourly'")
        
        interval = interval.strip().lower()

        # get the source origin time data series
        date_series = pd.to_datetime(self._clean_column('source_origin_time'))
        
        # start populate the origin time of data based on the given interval
        if interval == 'yearly':
            grouped = date_series.groupby(date_series.dt.year).size()
            x_values = grouped.index.astype(str)
            y_values = grouped.values
            x_label = "Yearly Intensities"
        elif interval == 'monthly':
            grouped = date_series.groupby([date_series.dt.year, date_series.dt.month]).size()
            x_values = [f"{year}_{month:02d}" for year, month in grouped.index]
            y_values = grouped.values
            x_label = "Monthly Intensities"
        elif interval == 'weekly':
            grouped = date_series.groupby([date_series.dt.isocalendar().year, date_series.dt.isocalendar().week]).size()
            x_values = [f"{year}_{week:02d}" for year, week in grouped.index]
            y_values = grouped.values
            x_label = "Weekly Intensities"
        elif interval == 'daily':
            grouped = date_series.groupby([date_series.dt.year, date_series.dt.month, date_series.dt.day]).size()
            x_values = [f"{year}_{month:02d}_{day:02d}" for year, month, day in grouped.index]
            y_values = grouped.values
            x_label = "Daily Intensities"
        elif interval == 'hourly':
            grouped = date_series.groupby([date_series.dt.year, date_series.dt.month, date_series.dt.day, date_series.dt.hour]).size()
            x_values = [f"{year}_{month:02d}_{day:02d} : {hour:02d}:--:--" for year, month, day, hour in grouped.index]
            y_values = grouped.values
            x_label = "Hourly Intensities"
        else:
            pass 

        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=f"Intensity",
                x=x_values,
                y=y_values,
                marker_color = 'skyblue',
            )
        )

        # Update layout
        fig.update_layout(
            width = plot_width,
            height = plot_height,
            title = f"Histogram of Earthquakes intensity ({interval.capitalize()})",
            xaxis_title= x_label,
            yaxis_title='Count',
            xaxis_tickangle=45,
            bargap = 0.2,
            showlegend=True,
            template = 'plotly_white',
            legend=dict(
                yanchor = "top",
                y = 0.99,
                xanchor = "right",
                x = 0.99,             
                bgcolor="rgba(255,255,255,0.5)"
                
            )
        )

        fig.show()

        return None
    

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
    
    Examples:
    ``` python
    >>> catalog_dir = r"lqt_catalog.csv"
    >>> lqt_ready = load_catalog(catalog_dir)
    >>> lqt_ready.head()
    """
    try:
        dataframe = load_data(catalog_file)
        if dataframe.empty:
            raise ValueError(f"Catalog file '{catalog_file}' is empty")
        return LqtAnalysis(dataframe)
    except (FileNotFoundError, ValueError) as e:
        raise type(e)(f"Failed to load '{catalog_file}': {str(e)} ") from e
