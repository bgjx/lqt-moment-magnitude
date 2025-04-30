"""
Configuration module for the lqt-moment-magnitude package.

Version: 0.2.1

This module defines the `CONFIG` singleton, which provides configuration parameters for
magnitude calculations, spectral fitting, and performance options. Configurations are
organized into three dataclasses: `MagnitudeConfig`, `SpectralConfig`, and `PerformanceConfig`.
Default values are defined in the dataclasses, but users can override them by providing a
`config.ini` file in the parent directory of this module.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

Examples:
    The `CONFIG` object is automatically loaded when the module is imported. To use the default
    configuration:

    ``` python
        from lqt_moment_magnitude.config import CONFIG
        print(CONFIG.wave.SNR_THRESHOLD)      # Access wave configuration
        print(CONFIG.spectral.F_MIN)          # Access spectral configuration
    ```

    To override the configuration, create a `config.ini` file in the parent working 
    directory with the following structure:

    ``` ini
        [Wave]
        resample_data = None
        snr_threshold = 2
        pre_filter = 0.001,0.005,55,60
        water_level = 60
        apply_post_instrument_removal_filter = True
        post_filter_f_min = 0.01
        post_filter_f_max = 30
        trim_method = dynamic
        sec_bf_p_arr_trim = 10
        sec_af_p_arr_trim = 50
        padding_before_arrival = 0.2
        min_p_window = 1.0
        max_p_window = 10.0
        min_s_window = 2.0
        max_s_window = 20.0
        noise_duration = 1.0
        noise_padding = 0.2

        [Magnitude]
        r_pattern_p = 0.52
        r_pattern_s = 0.63
        free_surface_factor = 2.0
        k_p = 0.32
        k_s = 0.21
        mw_constant = 6.07
        taup_model = iasp91
        velocity_model_file = velocity_model.json

        [Spectral]
        smooth_window_size = 3
        f_min = 0.01
        f_max = 30
        omega_0_range_min = 0.001
        omega_0_range_max = 2000
        q_range_min = 50
        q_range_max = 300
        default_n_samples = 3000
        n_factor = 2
        y_factor = 1

        [Performance]
        use_parallel = False
        logging_level = INFO
    ```

    For custom velocity model, the "velocity_model.json" file should have the 
    following structure:

    ``` json
        {
            "layer_boundaries": [[-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50]],
            "velocity_vp": [2.68, 2.99, 3.95, 4.50],
            "velocity_vs": [1.60, 1.79, 2.37, 2.69],
            "density": [2700, 2700, 2700, 2700]
        }
    ```

    You can also reload the configuration from a custom file:

    ``` python
        CONFIG.reload(config_file="new_config.ini")
    ```
"""

from importlib.resources import files, as_file
from contextlib import contextmanager
from dataclasses import dataclass
from configparser import ConfigParser
from typing import List, Tuple, Optional
from pathlib import Path
import json

@contextmanager
def _package_file(filename):
    """ Helper function to access package files using importlib.resources. """
    source = files("lqtmoment.data").joinpath(filename)
    with as_file(source) as file_path:
        yield file_path

@dataclass
class WaveConfig:
    """
    Configuration for wave treatment parameters.

    Attributes:
        RESAMPLE_DATA (float): New sampling rate value to be applied to seismogram data (default: None).
        SNR_THRESHOLD (float): Minimum signal-to-noise ratio for trace acceptance (default: 2).
        PRE_FILTER (List[float]): Bandpass filter corners [f1,f2,f3,f4] in Hz to be applied prior to 
                                    instrument response removal process (default: 0.001, 0.005, 55, 60).
        WATER_LEVEL (float): Water level for deconvolution stabilization during instrument removal (default: 60).
        APPLY_POST_INSTRUMENT_REMOVAL_FILTER (bool): If True, post filter after instrument removal will be applied (default: True).
        POST_FILTER_F_MIN (float): Minimum post-filter frequency in Hz (default: 0.01) after instrument removal.
        POST_FILTER_F_MAX (float): Maximum post-filter frequency in Hz (default: 30) after instrument removal.
        TRIM_MODE (str): Mode used for seismogram trimming. Defaults to dynamic, primarily using the coda information from the catalog.
        SEC_BF_P_ARR_TRIM (float): Time in seconds before P arrival as starting point of trimming (default: 10.0).
        SEC_AF_P_ARR_TRIM (float): Time in seconds after P arrival as ending point of trimming (default: 50.0).
        PADDING_BEFORE_ARRIVAL (float): Padding before arrival in seconds (default: 0.2).
        MIN_P_WINDOW (float): Minimum P phase window in second for calculating source spectra
                                (default: 1.0).
        MAX_P_WINDOW (float): Maximum P phase window in second for calculating source spectra
                                (default: 10.0).
        MIN_S_WINDOW (float): Minimum S phase window in second for calculating source spectra
                                (default: 2.0).
        MAX_S_WINDOW (float): Maximum S phase window in second for calculating source spectra
                                (default: 20.0).
        NOISE_DURATION (float): Noise window duration in seconds (default: 1.0).
        NOISE_PADDING (float): Noise window padding in seconds (default: 0.2).
    """
    RESAMPLE_DATA : float = None
    SNR_THRESHOLD: float = 2
    PRE_FILTER: List[float] = None
    WATER_LEVEL: float = 60    
    APPLY_POST_INSTRUMENT_REMOVAL_FILTER: bool = True
    POST_FILTER_F_MIN: float = 0.01
    POST_FILTER_F_MAX: float = 30.0
    TRIM_MODE: str = 'dynamic'
    SEC_BF_P_ARR_TRIM: float = 10.0
    SEC_AF_P_ARR_TRIM: float = 50.0
    PADDING_BEFORE_ARRIVAL: float = 0.2
    MIN_P_WINDOW: float = 1.0
    MAX_P_WINDOW: float = 10.0
    MIN_S_WINDOW: float = 2.0
    MAX_S_WINDOW: float = 20.0
    NOISE_DURATION: float = 1.0
    NOISE_PADDING: float = 0.2

    def __post_init__(self):
        self.PRE_FILTER = self.PRE_FILTER or [0.001, 0.005, 55, 60]

@dataclass
class MagnitudeConfig:
    """
    Configuration for magnitude calculation parameters.

    Attributes:
        R_PATTERN_P (float): Radiation pattern for P-waves (default: 0.52).
        R_PATTERN_S (float): Radiation pattern for S-waves (default: 0.63).
        FREE_SURFACE_FACTOR (float): Free surface amplification factor (default: 2.0).
        K_P (float): Geometric spreading factor for P-waves (default: 0.32).
        K_S (float): Geometric spreading factor for S-waves (default: 0.21).
        LAYER_BOUNDARIES (List[Tuple[float, float]]): Depth boundaries in km (default: placeholder).
        VELOCITY_VP (List[float]): P-wave velocities in km/s (default: placeholder).
        VELOCITY_VS (List[float]): S-wave velocities in km/s (default: placeholder).
        DENSITY (List[float]): Densities in kg/m³ (default: placeholder).
        TAUP_MODEL (str): ObsPy 1-D Velocity model.
        VELOCITY_MODEL_FILE (str): Path to a JSON file defining the velocity model (default: "", uses built-in model).
    """
    R_PATTERN_P: float = 0.52
    R_PATTERN_S: float = 0.63
    FREE_SURFACE_FACTOR: float = 2.0
    K_P: float = 0.32
    K_S: float = 0.21
    LAYER_BOUNDARIES: List[List[float]] = None 
    VELOCITY_VP: List[float] = None
    VELOCITY_VS: List[float] = None
    DENSITY: List[float] = None
    MW_CONSTANT: float = 6.07
    TAUP_MODEL: str = 'iasp91'
    VELOCITY_MODEL_FILE: str = None

    def __post_init__(self):
        self.LAYER_BOUNDARIES = self.LAYER_BOUNDARIES or [
                [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50],
                [2.50, 7.00], [7.00, 9.00], [9.00, 15.00], [15.00, 33.00], [33.00, 9999]
            ]
        self.VELOCITY_VP = self.VELOCITY_VP or [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
        self.VELOCITY_VS = self.VELOCITY_VS or [1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79]
        self.DENSITY = self.DENSITY or [2700] * 9

        # Load velocity model from a defautl package JSON data
        if self.VELOCITY_MODEL_FILE == 'None' or self.VELOCITY_MODEL_FILE is None:
            try:
                with _package_file("velocity_model.json") as velocity_model_path:
                    with open(velocity_model_path, "r") as f:
                        model = json.load(f)
                    required_keys = {"layer_boundaries", "velocity_vp", "velocity_vs", "density"}
                    if not all(key in model for key in required_keys):
                        raise KeyError(f"Missing keys: {required_keys - set(model.keys())}")
                    self.LAYER_BOUNDARIES = model["layer_boundaries"]
                    self.VELOCITY_VP = model["velocity_vp"]
                    self.VELOCITY_VS = model["velocity_vs"]
                    self.DENSITY = model["density"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Failed to load velocity model: {e}. Using defaults")
                    
        else:
            # Load from user-specified file
            try:
                with open(Path(self.VELOCITY_MODEL_FILE), "r") as f:
                    model = json.load(f)
                required_keys = {"layer_boundaries", "velocity_vp", "velocity_vs", "density"}
                if not all(key in model for key in required_keys):
                    raise KeyError(f"Missing keys: {required_keys - set(model.keys())}")
                self.LAYER_BOUNDARIES = model["layer_boundaries"]
                self.VELOCITY_VP = model["velocity_vp"]
                self.VELOCITY_VS = model["velocity_vs"]
                self.DENSITY = model["density"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Failed to load velocity model: {e}. Using defaults.")

        # Validation
        if not(len(self.LAYER_BOUNDARIES) == len(self.VELOCITY_VP) == len(self.VELOCITY_VS) == len(self.DENSITY)):
            raise ValueError("LAYER_BOUNDARIES, VELOCITY_VP, VELOCITY_VS, and DENSITY must have the same length")
        if any(vp <= 0 for vp in self.VELOCITY_VP) or any(vs <= 0 for vs in self.VELOCITY_VS):
            raise ValueError("Velocities must be positive")
        if any(d <= 0 for d in self.DENSITY):
            raise ValueError("Densities must be positive")
        if self.R_PATTERN_P <= 0 or self.R_PATTERN_S <= 0:
            raise ValueError("R_PATTERN_P and R_PATTERN_S must be positive")
        if self.K_P <= 0 or self.K_S <= 0:
            raise ValueError("K_P and K_S must be positive")

@dataclass
class SpectralConfig:
    """
    Configuration for spectral fitting parameters.
    
    Attributes:
        SMOOTH_WINDOW_SIZE (int): Size of the moving average window for smoothing,
                                    must be odd positive, if None no smoothing applied (default: 3).
        F_MIN (float): Minimum frequency for fitting in Hz (default: 0.01).
        F_MAX (float): Maximum frequency for fitting in Hz (default: 30.0).
        OMEGA_0_RANGE_MIN (float): Minimum Omega_0 in nm/Hz (default: 0.01).
        OMEGA_0_RANGE_MAX (float): Maximum Omega_0 in nm/Hz (default: 2000.0).
        Q_RANGE_MIN (float): Minimum quality factor Q (default: 50.0).
        Q_RANGE_MAX (float): Maximum quality factor Q (default: 300.0).
        DEFAULT_N_SAMPLES (int): Default number for stochastic random sampling (default: 3000).
        N_FACTOR (int): Brune model n factor for spectral decay (default: 2).
        Y_FACTOR (int): Brune model y factor for spectral decay (default: 1).
    """
    SMOOTH_WINDOW_SIZE: int = 3
    F_MIN: float = 0.01
    F_MAX: float = 30
    OMEGA_0_RANGE_MIN: float = 0.01
    OMEGA_0_RANGE_MAX: float = 2000.0
    Q_RANGE_MIN: float = 50.0
    Q_RANGE_MAX: float = 300.0
    DEFAULT_N_SAMPLES: int = 3000
    N_FACTOR: int = 2
    Y_FACTOR: int = 1


@dataclass
class PerformanceConfig:
    """
    Configuration for performance options.

    Attributes:
        USE_PARALLEL (bool): Enable parallel processing (default: False)
        LOGGING_LEVEL (str): Logging verbosity (DEBUG, INFO, WARNING, ERROR, default: INFO)

    """
    USE_PARALLEL: bool = False
    LOGGING_LEVEL: str = "INFO"


class Config:
    """
    A config class for combines magnitude, spectral, and performance configurations with loading from INI file.

    The configuration is loaded from a `config.ini` file, with fallback to defaults if the file
    or specific parameters are not found.

    """
    def __init__(self):
        self.wave = WaveConfig()
        self.magnitude = MagnitudeConfig()
        self.spectral = SpectralConfig()
        self.performance = PerformanceConfig()

    
    def _parse_float(self, config_section, key, fallback):
        """
        Parsing method for float values from config with validation.
        
        Args:
            config_section: ConfigParser section object to parse from.
            key (str): Key to parse.
            fallback: Fallback value if key is not found.
        
        Returns: 
            float or None: Parsed float value or None if specified.
        
        Raises:
            ValueError: If the value cannot be parsed as a float.     
        """
        raw_value = config_section.get(key, fallback=str(fallback))
        if raw_value is None or raw_value.strip().lower() in ['none', '']:
            return None
        
        try:
            value = float(raw_value)
            return value
        except ValueError as e:
            raise ValueError(f"Invalid float for {key} in config.ini: {e}")
    

    def _parse_int(self, config_section, key, fallback):
        """
        Parsing method for int values from config with validation.
        
        Args:
            config_section: ConfigParser section object to parse from.
            key (str): key to parse.
            fallback: Fallback value if key is not found.
        
        Returns:
            int or None: Parsed integer value or None if specified.
        
        Raises:
            ValueError: If the value cannot be parsed as an integer.
        """
        raw_value = config_section.get(key, fallback=str(fallback))
        if raw_value is None or raw_value.strip().lower() in ['none', '']:
            return None
        try:
            value = int(raw_value)
            return value
        except ValueError as e:
            raise ValueError(f"Invalid int for {key} in config.ini: {e}")
    

    def _parse_list(self, config_section, key, fallback, delimiter=","):
        """
        Parsing method for list values from config with validation.
        
        Args:
            config_section: ConfigParser section object to parse from.
            key (str): Key to Parse.
            fallback: Fallback value if key is not found.
            delimiter (str): Delimiter to split the string (default: ",").
        
        Returns:
            List[float]: List of parsed float values.
        
        Raises:
            ValueError: If the value cannot be parsed as a list of floats.
        """
        try:
            return [float(x) for x in config_section.get(key, fallback=fallback).split(delimiter)]
        except ValueError as e:
            raise ValueError(f"Invalid format for {key} in config.ini: {e}")


    def load_from_file(self, config_file: Optional[str] = None) -> None:
        """
        Load configuration from an INI file, with fallback to defaults.
        
        Args:
            config_file (Optional[str]): Path to configuration file.
            Defaults to 'config.ini' in parent directory.
        
        Raises:
            FileNotFoundError: If the configuration file is not found or unreadable.
            ValueError: If configuration parameters are invalid.       
        """
        config  = ConfigParser()
        if config_file is None:
            # Load the default config.ini from package default data
            with _package_file("config.ini") as default_config_path:
                if not config.read(default_config_path):
                    raise FileNotFoundError(f"Default configuration file {default_config_path} not found in package")
        else:
            config_file = Path(config_file)
            if not config.read(config_file):
                raise FileNotFoundError(f"Configuration file {config_file} not  found or unreadable")
        
        # Load wave config section
        if "Wave" in config:
            wave_section = config["Wave"]
            resample_data = self._parse_float(wave_section, "resample", self.wave.RESAMPLE_DATA)
            if resample_data is not None and resample_data <= 0:
                raise ValueError("new sampling rate must be positive value or None.")
            snr_threshold = self._parse_float(wave_section, "snr_threshold", self.wave.SNR_THRESHOLD)
            if snr_threshold <= 0:
                raise ValueError("snr_threshold must be positive")
            pre_filter = self._parse_list(wave_section, "pre_filter", "0.01,0.02,55,60")
            if len(pre_filter) != 4 or any(f <=0 for f in pre_filter):
                raise ValueError("pre_filter must be four positive frequencies (f1, f2, f3, f4)")
            water_level = self._parse_float(wave_section, "water_level", self.wave.WATER_LEVEL)
            if water_level is not None and water_level < 0:
                raise ValueError("water_level must be non negative or None, otherwise mathematically meaningless")
            apply_post_instrument_removal_filter = wave_section.getboolean("apply_post_instrument_removal_filter", fallback=self.wave.APPLY_POST_INSTRUMENT_REMOVAL_FILTER)
            post_filter_f_min = self._parse_float(wave_section, "post_filter_f_min", self.wave.POST_FILTER_F_MIN)
            if post_filter_f_min < 0:
                raise ValueError("post_filter_f_min must be non negative value")
            post_filter_f_max = self._parse_float(wave_section, "post_filter_f_max", self.wave.POST_FILTER_F_MAX)
            if post_filter_f_max <= post_filter_f_min:
                raise ValueError("post_filter_f_max must be greater than post_filter_f_min")
            trim_mode = wave_section.get("trim_mode", fallback=self.wave.TRIM_MODE)
            if trim_mode not in ['dynamic', 'static']:
                raise ValueError("trim method must be either 'dynamic' or 'static'")
            sec_bf_p_arr_trim = self._parse_float(wave_section, "sec_bf_arr_trim", self.wave.SEC_BF_P_ARR_TRIM)
            if sec_bf_p_arr_trim < 0:
                 raise ValueError("Time before P arrival for trimming must be non-negative value")
            sec_af_p_arr_trim = self._parse_float(wave_section, "sec_af_arr_trim", self.wave.SEC_AF_P_ARR_TRIM)
            if sec_af_p_arr_trim <= sec_bf_p_arr_trim:
                 raise ValueError("Time after P arrival for trimming must be greater than the time before.")
            padding_before_arrival = self._parse_float(wave_section, "padding_before_arrival", self.wave.PADDING_BEFORE_ARRIVAL)
            if padding_before_arrival < 0:
                raise ValueError("padding_before_arrival must be non-negative")
            min_p_window = self._parse_float(wave_section, "min_p_window", self.wave.MIN_P_WINDOW)
            if min_p_window <= 0:
                raise ValueError("min_p_window must be positive")
            max_p_window = self._parse_float(wave_section, "max_p_window", self.wave.MAX_P_WINDOW)
            if max_p_window < min_p_window:
                raise ValueError("max_p_window must be greater than min_p_window")
            min_s_window = self._parse_float(wave_section, "min_s_window", self.wave.MIN_S_WINDOW)
            if min_s_window <= 0:
                raise ValueError("min_s_window must be positive")
            max_s_window = self._parse_float(wave_section, "max_s_window", self.wave.MAX_S_WINDOW)
            if max_s_window < min_s_window:
                raise ValueError("max_s_window must be greater than min_s_window")
            noise_duration = self._parse_float(wave_section, "noise_duration", self.wave.NOISE_DURATION)
            if noise_duration <= 0:
                raise ValueError("noise_duration must be positive")
            noise_padding = self._parse_float(wave_section, "noise_padding", self.wave.NOISE_PADDING)
            if noise_padding < 0:
                raise ValueError("noise_padding must be non-negative")

            # Reconstruct WaveConfig to trigger __post_init__
            self.wave = WaveConfig(
                RESAMPLE_DATA=resample_data,
                SNR_THRESHOLD=snr_threshold,
                PRE_FILTER=pre_filter,
                WATER_LEVEL=water_level,
                APPLY_POST_INSTRUMENT_REMOVAL_FILTER=apply_post_instrument_removal_filter,
                POST_FILTER_F_MIN=post_filter_f_min,
                POST_FILTER_F_MAX=post_filter_f_max,
                TRIM_MODE=trim_mode,
                SEC_BF_P_ARR_TRIM=sec_bf_p_arr_trim,
                SEC_AF_P_ARR_TRIM=sec_af_p_arr_trim,
                PADDING_BEFORE_ARRIVAL=padding_before_arrival,
                MIN_P_WINDOW=min_p_window,
                MAX_P_WINDOW=max_p_window,
                MIN_S_WINDOW=min_s_window,
                MAX_S_WINDOW=max_s_window,
                NOISE_DURATION=noise_duration,
                NOISE_PADDING=noise_padding
            )
            
        # Load magnitude config section
        if "Magnitude" in config:
            mag_section = config["Magnitude"]
            r_pattern_p = self._parse_float(mag_section, "r_pattern_p", self.magnitude.R_PATTERN_P)
            r_pattern_s = self._parse_float(mag_section, "r_pattern_s", self.magnitude.R_PATTERN_S)
            free_surface_factor = self._parse_float(mag_section, "free_surface_factor", self.magnitude.FREE_SURFACE_FACTOR)
            if free_surface_factor <= 0:
                raise ValueError("free_surface_factor must be positive")
            k_p = self._parse_float(mag_section, "k_p", self.magnitude.K_P)
            k_s = self._parse_float(mag_section, "k_s", self.magnitude.K_S)
            mw_constant = self._parse_float(mag_section, "mw_constant",self.magnitude.MW_CONSTANT)
            taup_model = mag_section.get("taup_model", fallback=self.magnitude.TAUP_MODEL)
            velocity_model_file = mag_section.get("velocity_model_file", fallback=self.magnitude.VELOCITY_MODEL_FILE)
            
            # Reconstruct MagnitudeConfig to trigger __post_init__
            self.magnitude = MagnitudeConfig(
                R_PATTERN_P=r_pattern_p,
                R_PATTERN_S=r_pattern_s,
                FREE_SURFACE_FACTOR=free_surface_factor,
                K_P=k_p,
                K_S=k_s,
                MW_CONSTANT=mw_constant,
                TAUP_MODEL=taup_model,
                VELOCITY_MODEL_FILE=velocity_model_file    
            )

            # Validate TAUP_MODEL
            from obspy.taup import TauPyModel
            try:
                TauPyModel(model=self.magnitude.TAUP_MODEL)
            except Exception as e:
                raise ValueError(f"Invalid taup_model '{self.magnitude.TAUP_MODEL}': {e}")

        # Load spectral config section
        if "Spectral" in config:
            spec_section = config["Spectral"]
            self.spectral.SMOOTH_WINDOW_SIZE = self._parse_int(spec_section, "smooth_window_size", self.spectral.SMOOTH_WINDOW_SIZE)
            if (self.spectral.SMOOTH_WINDOW_SIZE is not None and self.spectral.SMOOTH_WINDOW_SIZE % 2 == 0) or (self.spectral.SMOOTH_WINDOW_SIZE is not None and self.spectral.SMOOTH_WINDOW_SIZE < 0) :
                raise ValueError("smooth_window_size must be odd positive value or None")
            self.spectral.F_MIN = self._parse_float(spec_section, "f_min", self.spectral.F_MIN)
            if self.spectral.F_MIN <= 0:
                raise ValueError("f_min must be positive")
            self.spectral.F_MAX = self._parse_float(spec_section, "f_max", self.spectral.F_MAX)
            if self.spectral.F_MAX < self.spectral.F_MIN:
                raise ValueError("f_max must be greater than f_min")
            self.spectral.OMEGA_0_RANGE_MIN = self._parse_float(spec_section, "omega_0_range_min", self.spectral.OMEGA_0_RANGE_MIN)
            if self.spectral.OMEGA_0_RANGE_MIN <= 0:
                raise ValueError("omega_0_range_min must be positive")
            self.spectral.OMEGA_0_RANGE_MAX = self._parse_float(spec_section, "omega_0_range_max", self.spectral.OMEGA_0_RANGE_MAX)
            if self.spectral.OMEGA_0_RANGE_MAX <= self.spectral.OMEGA_0_RANGE_MIN:
                raise ValueError("omega_0_range_max must be greater than omega_0_range_min")
            self.spectral.Q_RANGE_MIN = self._parse_float(spec_section, "q_range_min", self.spectral.Q_RANGE_MIN)
            if self.spectral.Q_RANGE_MIN <= 0:
                raise ValueError("q_range_min must be positive")
            self.spectral.Q_RANGE_MAX = self._parse_float(spec_section, "q_range_max", self.spectral.Q_RANGE_MAX)
            if self.spectral.Q_RANGE_MAX <= self.spectral.Q_RANGE_MIN:
                raise ValueError("q_range_max must be greater than q_range_min")
            self.spectral.DEFAULT_N_SAMPLES = self._parse_int(spec_section, "default_n_samples", self.spectral.DEFAULT_N_SAMPLES)
            if self.spectral.DEFAULT_N_SAMPLES <= 0:
                raise ValueError("default_n_samples must be positive")
            self.spectral.N_FACTOR = self._parse_int(spec_section, "n_factor", self.spectral.N_FACTOR)
            self.spectral.Y_FACTOR = self._parse_int(spec_section, "y_factor", self.spectral.Y_FACTOR)
        
        # Load performance config section
        if "Performance" in config:
            perf_section = config["Performance"]
            self.performance.USE_PARALLEL = perf_section.getboolean("use_parallel", fallback=self.performance.USE_PARALLEL)
            self.performance.LOGGING_LEVEL = perf_section.get("logging_level", fallback=self.performance.LOGGING_LEVEL)
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            if self.performance.LOGGING_LEVEL not in valid_levels:
                raise ValueError(f"logging_level must be one of: {valid_levels}")
    
    def reload(self, config_file:Optional[str] = None) -> None:
        """
        Reload configuration from INI file, resetting to defaults first.

        Args:
            config_file (Optional[str]): Path to the configuration file.
        """
        self.__init__()
        self.load_from_file(config_file)

# Singleton instance for easy access
CONFIG = Config()
CONFIG.load_from_file()