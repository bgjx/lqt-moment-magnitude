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
from typing import Optional, List, Callable, Tuple
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
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")
            if dataframe.empty:
                raise ValueError("DataFrame cannot be empty")
        self.data = dataframe
    






    

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
