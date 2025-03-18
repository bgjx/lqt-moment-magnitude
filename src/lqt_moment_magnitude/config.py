from dataclasses import dataclass
from configparser import ConfigParser
from typing import List, Tuple
from pathlib import Path

@dataclass
class MagnitudeConfig:
    """
    
    Configuration for magnitude calculation parameters.

    Attributes:
        SNR_THRESHOLD (float): Minimum signal-to-noise ratio for trace acceptance (default: 1.5).
        WATER_LEVEL (int): Water level for deconvolution stabilization (default: 30).
        PRE_FILTER (List[float]): Bandpass filter corners [f1,f2,f3,f4] in Hz (default: placeholder, override in config.ini).
        POST_FILTER_F_MIN (float): Minimum post-filter frequency in Hz (default: 0.1).
        POST_FILTER_F_MAX (float): Maximum post-filter frequency in Hz (default: 50).
        PADDING_BEFORE_ARRIVAL (float): Padding before arrival in seconds (default: 0.1).
        NOISE_DURATION (float): Noise window duration in seconds (default: 0.5).
        NOISE_PADDING (float): Noise window padding in seconds (default: 0.2).
        R_PATTERN_P (float): Radiation pattern for P-waves (default: 0.52).
        R_PATTERN_S (float): Radiation pattern for S-waves (default: 0.63).
        FREE_SURFACE_FACTOR (float): Free surface amplification factor (default: 2.0).
        K_P (float): Geometric spreading factor for P-waves (default: 0.32).
        K_S (float): Geometric spreading factor for S-waves (default: 0.21).
        LAYER_BOUNDARIES (List[Tuple[float, float]]): Depth boundaries in km (default: placeholder).
        VELOCITY_VP (List[float]): P-wave velocities in km/s (default: placeholder).
        VELOCITY_VS (List[float]): S-wave velocities in km/s (default: placeholder).
        DENSITY (List[float]): Densities in kg/m³ (default: placeholder). 
    """
    SNR_THRESHOLD: float = 1.5
    WATER_LEVEL: int = 30
    PRE_FILTER: List[float] = None
    POST_FILTER_F_MIN: float = 0.1
    POST_FILTER_F_MAX: float = 50
    PADDING_BEFORE_ARRIVAL: float = 0.1
    NOISE_DURATION: float = 0.5
    NOISE_PADDING: float = 0.2
    R_PATTERN_P: float = 0.52
    R_PATTERN_S: float = 0.63
    FREE_SURFACE_FACTOR: float = 2.0
    K_P: float = 0.32
    K_S: float = 0.21
    LAYER_BOUNDARIES: List[List[float]] = None 
    VELOCITY_VP: List[float] = None
    VELOCITY_VS: List[float] = None
    DENSITY: List[float] = None


    def __post_init__(self):
        if self.PRE_FILTER is None:
            self.PRE_FILTER = [0.01, 0.02, 55, 60]
        if self.LAYER_BOUNDARIES is None:
            self.LAYER_BOUNDARIES = [
                [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50],
                [2.50, 7.00], [7.00, 9.00], [9.00, 15.00], [15.00, 33.00], [33.00, 9999]
            ]
        if self.VELOCITY_VP is None:
            self.VELOCITY_VP = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
        if self.VELOCITY_VS is None:
            self.VELOCITY_VS = [1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79]
        if self.DENSITY is None:
            self.DENSITY = [2700] * 9

@dataclass
class SpectralConfig:
    """
    Configuration for spectral fitting parameters.
    
    Attributes:
        F_MIN (float): Minimum frequency for fitting in Hz (default: 1.0).
        F_MAX (float): Maximum frequency for fitting in Hz (default: 45.0).
        OMEGA_0_RANGE_MIN (float): Minimum Omega_0 in meters (default: 0.01).
        OMEGA_0_RANGE_MAX (float): Maximum Omega_0 in meters (default: 2000.0).
        Q_RANGE_MIN (float): Minimum quality factor Q (default: 50.0).
        Q_RANGE_MAX (float): Maximum quality factor Q (default: 250.0).
        FC_RANGE_BUFFER (float): Buffer factor for corner frequency range (default: 2.0).
        DEFAULT_N_SAMPLES (int): Default number for stochastic random sampling (default: 3000).
        N_FACTOR (int): Brune model n factor (default: 2).
        Y_FACTOR (int): Brune model y factor (default: 1).
    """
    F_MIN: float = 1.0
    F_MAX: float = 45.0
    OMEGA_0_RANGE_MIN: float = 0.01
    OMEGA_0_RANGE_MAX: float = 2000.0
    Q_RANGE_MIN: float = 50.0
    Q_RANGE_MAX: float = 250.0
    FC_RANGE_BUFFER: float = 2.0
    DEFAULT_N_SAMPLES: int = 3000
    N_FACTOR: int = 2
    Y_FACTOR: int = 1

@dataclass
class PerformanceConfig:
    """
    Configuration for performance options.

    Attributes:
        USE_PARALLEL (bool): Enable parallel processing (default: False)

    """
    USE_PARALLEL: bool = False

class Config:
    """ Combines magnitude, spectral, and performance configurations with loading from INI file. """
    def __init__(self):
        self.magnitude = MagnitudeConfig()
        self.spectral = SpectralConfig()
        self.performance = PerformanceConfig()

    def load_from_file(self, config_file: str = None) -> None:
        """
        Load configuration from an INI file, with fallback to defaults.
        
        Args:
            config_file (str, optional): Path to configuration file.
            Defaults to 'config.ini' in parent directory.
        
        Raises:
            FileNotFoundError: If the configuration file is not found or unreadable.
            ValueError: If configuration parameters are invalid.       
        """
        config  = ConfigParser()
        if config_file is None:
            config_file = Path(__file__).parent.parent/ "config.ini"
        if not config.read(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found or unreadable")
        
        # Load magnitude config section
        if "Magnitude" in config:
            mag_section = config["Magnitude"]
            self.magnitude.SNR_THRESHOLD = mag_section.getfloat("snr_threshold", fallback=self.magnitude.SNR_THRESHOLD)
            self.magnitude.WATER_LEVEL = mag_section.getint("water_level", fallback=self.magnitude.WATER_LEVEL)
            try:
                self.magnitude.PRE_FILTER = [float(x) for x in mag_section.get("pre_filter", fallback="2,5,55,60").split(",")]
            except ValueError as e:
                raise ValueError(f"Invalid format for pre_filter in config.ini: {e}")
            self.magnitude.POST_FILTER_F_MIN = mag_section.getfloat("post_filter_f_min", fallback=self.magnitude.POST_FILTER_F_MIN)
            self.magnitude.POST_FILTER_F_MAX = mag_section.getfloat("post_filter_f_max", fallback=self.magnitude.POST_FILTER_F_MAX)
            self.magnitude.PADDING_BEFORE_ARRIVAL = mag_section.getfloat("padding_before_arrival", fallback=self.magnitude.PADDING_BEFORE_ARRIVAL)
            self.magnitude.NOISE_DURATION = mag_section.getfloat("noise_duration", fallback=self.magnitude.NOISE_DURATION)
            self.magnitude.NOISE_PADDING = mag_section.getfloat("noise_padding", fallback=self.magnitude.NOISE_PADDING)
            self.magnitude.R_PATTERN_P = mag_section.getfloat("r_pattern_p", fallback=self.magnitude.R_PATTERN_P )
            self.magnitude.R_PATTERN_S = mag_section.getfloat("r_pattern_s", fallback=self.magnitude.R_PATTERN_S)
            self.magnitude.FREE_SURFACE_FACTOR = mag_section.getfloat("free_surface_factor", fallback=self.magnitude.FREE_SURFACE_FACTOR)
            self.magnitude.K_P = mag_section.getfloat("k_p", fallback=self.magnitude.K_P)
            self.magnitude.K_S = mag_section.getfloat("k_s", fallback=self.magnitude.K_S)
            boundaries_str = mag_section.get("layer_boundaries", fallback= "-3.00,-1.90; -1.90,-0.59; -0.59, 0.22; 0.22, 2.50; 2.50, 7.00; 7.00,9.00;  9.00,15.00 ; 15.00,33.00; 33.00,9999")
            try:
                self.magnitude.LAYER_BOUNDARIES = [[float(x) for x in layer.split(",")] for layer in boundaries_str.split(";")]
            except ValueError as e:
                raise ValueError(f"Invalid format for layer_boundaries i config.ini: {e}")
            try:
                self.magnitude.VELOCITY_VP = [float(x) for x in mag_section.get("velocity_vp", fallback="2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00").split(",")]
                self.magnitude.VELOCITY_VS = [float(x) for x in mag_section.get("velocity_vs", fallback="1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79").split(",")]
                self.magnitude.DENSITY = [float(x) for x in mag_section.get("density", fallback="2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700").split(",")]
            except ValueError as e:
                raise ValueError(f"Invalid format for velocity or density in config.ini: {e}")

        # Load spectral config section
        if "Spectral" in config:
            spec_section = config["Spectral"]
            self.spectral.F_MIN = spec_section.getfloat("f_min", fallback=self.spectral.F_MIN)
            self.spectral.F_MAX = spec_section.getfloat("f_max", fallback=self.spectral.F_MAX)
            self.spectral.OMEGA_0_RANGE_MIN = spec_section.getfloat("omega_0_range_min", fallback=self.spectral.OMEGA_0_RANGE_MIN)
            self.spectral.OMEGA_0_RANGE_MAX = spec_section.getfloat("omega_0_range_max", fallback=self.spectral.OMEGA_0_RANGE_MAX)
            self.spectral.Q_RANGE_MIN =  spec_section.getfloat("q_range_min", fallback=self.spectral.Q_RANGE_MIN)
            self.spectral.Q_RANGE_MAX =  spec_section.getfloat("q_range_max", fallback=self.spectral.Q_RANGE_MAX)
            self.spectral.FC_RANGE_BUFFER = spec_section.getfloat("fc_range_buffer", fallback=self.spectral.FC_RANGE_BUFFER)
            self.spectral.DEFAULT_N_SAMPLES = spec_section.getint("default_n_samples", fallback=self.spectral.DEFAULT_N_SAMPLES)
            self.spectral.N_FACTOR = spec_section.getint("n_factor", fallback=self.spectral.N_FACTOR)
            self.spectral.Y_FACTOR = spec_section.getint("y_factor", fallback=self.spectral.Y_FACTOR)
        
        # Load performance config section
        if "Performance" in config:
            perf_section = config["Performance"]
            self.performance.USE_PARALLEL = perf_section.getboolean("use_parallel", fallback=self.performance.USE_PARALLEL)
    
    def reload(self, config_file: str = None) -> None:
        """
        Reload configuration from INI file, resetting to defaults first.

        Args:
            config_file (str, optional): Path to the configuration file.
        """

        self.__init__()
        self.load_from_file(config_file)

# Singleton instance for easy access
CONFIG = Config()
CONFIG.load_from_file()