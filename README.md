[![Follow me on Twitter](https://img.shields.io/badge/Follow-@zakki_edelo-blue?logo=x&logoColor=white&style=flat)](https://twitter.com/zakki_edelo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arham_zakki-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/arhamzakki)
[![CI - Tests](https://github.com/bgjx/lqt-moment-magnitude/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/bgjx/lqt-moment-magnitude/actions/workflows/ci-cd.yml)
[![GitHub Issues](https://img.shields.io/github/issues/bgjx/lqt-moment-magnitude?style=flat)](https://github.com/bgjx/lqt-moment-magnitude/issues)
[![GitHub Commits](https://img.shields.io/github/last-commit/bgjx/lqt-moment-magnitude?style=flat)](https://github.com/bgjx/lqt-moment-magnitude/commits/main/)
[![PyPI](https://img.shields.io/pypi/v/lqtmoment?style=flat$logo=pypi)](https://pypi.org/project/lqtmoment/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org/)


## What is it?

**lqtmoment** is python package designed for moment magnitude calculations using pure P, SV, and SH components in the LQT ray coordinate system. It leverages rapid ray tracing to compute incidence angles for component rotation and employs fast spectral fitting to find optimal solution, ensuring high accuracy and efficient automated computation.

By leveraging vectorized computing and advanced statistical methods—such as in implementing Shooting Snell’s Method for incidence angle estimation and Quasi-Monte Carlo techniques for spectral fitting—lqtmoment excels at calculating moment magnitudes for batches of earthquakes, handling hundreds to thousands of events in a single run.

--------------
### **Lqtmoment Test with Real Data**

Below is the `lqtmoment` test using real data, moment magnitudes for `700++ earthquakes` were successfully calculated in `LQT ray coordinate systems`, with an average of ~1.8 seconds per iteration. The seismogram data was recorded at `500 sps` using a 3-component shallow borehole sensor in 15 stations network.

``` python

    # lqtmoment test
    from lqtmoment import magnitude_estimator
    from pathlib import Path

    # directory object
    dirs = {"wave_dir": r"test\wave",
            "calib_dir": r"test\calibration",
            "catalog_file": r"lqt_catalog.csv",
            "config_file": r"config.ini"    
    }

    merged_lqt_catalog, lqt_moment_result, lqt_fitting_result = magnitude_estimator(    
                                                                wave_dir= dirs['wave_dir'],
                                                                cal_dir= dirs['calib_dir'],
                                                                catalog_file= dirs['catalog_file'],
                                                                config_file= dirs['config_file'],
                                                                id_start=2000,
                                                                id_end=2795,
                                                                lqt_mode=True,
                                                                generate_figure=False
                                                                )
```
``` bash
    Processing earthquakes: 100%|███████| 796/796 [07:31<00:00,  1.76it/s, Failed=0]
    Finished. Proceed 796 earthquakes successfully,0 failed. Check lqt_runtime.log for details. 

```

--------------

The **lqtmoment** includes modules for building input catalog format, performing moment magnitude calculation, visualizations, and data analysis.

Contact the developer: Arham Zakki Edelo (edelo.arham@gmail.com)

* [Installation](#Installation)
* [Tutorials](#Tutorials)
* [Scope of Capabilities](#Scope-of-Capabilities)
* [Examples](#Examples)
* [References](#References)
* [Contributing](#Contributing)
* [Report Bugs](#Report-Bugs)

--------------
### Installation
**lqtmoment** can be installed and run on multiple platforms, including macOS, Windows, and Linux, using Python versions 3.9 to 3.12. Choose one of the following installation methods:

#### Option 1: Via Anaconda
If you have Anaconda installed, create and activate a new environment (for clean installation), then install **lqtmoment** from the `bgjx` channel:

```bash
    conda create -n lqtmoment python=3.9
    conda activate lqtmoment
    conda install -c bgjx lqtmoment
```

#### Option 2: Via PyPI
It'is recommended (but optional) to upgrade `pip` first. Ensure you're in a virtual environment if desired.

```bash
    python -m pip install --upgrade pip
    python -m pip install lqtmoment

```

#### Option 3: Build from Source Code

To build from source, you need `Git` to be installed first in your computer and then you can clone the source from `GitHub` repo:

```bash
    git clone https://github.com/bgjx/lqt-moment-magnitude.git
```

Navigate to the package directory, to `lqt-moment-magnitude`

```bash
    cd lqt-moment-magnitude
```

And install the package

```bash
    python -m pip install .
```

--------------
### Tutorials

A series of tutorials are provided here: 

[Tutorials](https://github.com/bgjx/lqt-moment-magnitude/tree/main/lqt-tutorials)

These tutorials explain how to prepare input data, set initial parameters for your specific case, run the moment magnitude calculation, and use other features. 

--------------
### Scope of Capabilities
Below are key details outlining the capabilities and limitations of `lqtmoment` in its current version:

- **1. Earthquake Classification**: 
    **lqtmoment** automatically categorizes earthquake data based on epicentral distance or source depth:
    - `very_local_earthquake`: Epicentral distance < 30 km or < 2× source depth (whichever is satisfied first).
    - `local_earthquake`: Epicentral distance > 30 km or > 2× source depth (whichever is satisfied first) and < 100 km.
    - `regional_earthquake`: Epicentral distance > 100 km and < 1110 km.
    - `far_regional_earthquake`: Epicentral distance > 1110 km and < 2220 km.
    - `teleseismic_earthquake`: Epicentral distance > 2220 km.

- **2. Velocity Model Usage**:
    In this current version, user defined velocity model (in a `.json` file) is applied only for calculating the incidence angle via ray tracing (Shooting Snell's Method) for ZNE to LQT rotation for `very_local_earthquake` and `local_earthquake` classifications. For farther earthquake types,  the model defaults to `TauPyModel` from `obspy.taup` module configurable in `config.ini`

- **3. Velocity Model Limitation**:
    For this current version, the user-defined velocity model (e.g., `velocity_model.json`, or any name you choose), still and must be a 1-D velocity model for rapid estimation.

- **4. Incidence Angle Calculation**:
    Incidence angles are computed using ray tracing of refracted P and S waves, enabling ZNE-to-LQT rotation across all earthquake classifications:  
    - For `very_local_earthquake` and `local_earthquake`, **lqtmoment** uses an internal method: it performs vectorized computation based on Shooting Snell's Method and energy comparison between direct (Pg/Sg) and critically refracted (Pn/Sn) phases, selecting the incidence angle of the stronger phase if their arrival times differ by at least ~2× the dominant period. This ensures robustness, rapidness and precision for local events where refracted waves dominate. 
    - For farther earthquake classifications, incidence angles for P and S phases are retrieved from the TauPyModel (via obspy.taup), supporting LQT rotation without energy-based phase selection.
    - Reflected waves (e.g., pP, sP) are not currently considered, which may affect accuracy for shallow events (<10 km depth) or teleseismic distances (>2220 km) where reflections contribute significant energy. Future versions may support reflected waves and extended internal ray tracing optionally.

- **4. Testing Status**:
    Due to limited datasets, **lqtmoment** has been rigorously tested only for `very_local_earthquake` and `local_earthquake` classifications. If you encounter miscalculations in other earthquake categoris,  please report them as issues here: [Report Issues](https://github.com/bgjx/lqt-moment-magnitude/issues), any support will be beneficial for future development.


--------------
### Examples

**lqtmoment** supports two ways to run it: a **Programmatic Approach** for integration with your Python code or data analysis workflows, and a simpler **Command-Line Interface (CLI)** for straightforward usage:

**1. Programmatic Approach**:
```python
    from lqtmoment import magnitude_estimator
    calculated_moment_results, detailed_fitting_results = magnitude_estimator(
                                                        wave_dir = "..\tests\sample_tests_data\data\wave",
                                                        cal_dir = "..\tests\sample_tests_data\data\calibration",
                                                        catalog_file = "..\tests\sample_tests_data\results\lqt_catalog\lqt_catalog.xlsx",
                                                        config_file = "..\tests\sample_tests_data\calculation configuration\config_test.ini",
                                                        fig_dir = "..\tests\sample_tests_data\figures",
                                                        output_dir = "..\tests\sample_tests_data\results\calculation",
                                                        id_start = 1001,
                                                        id_end = 1005,
                                                        lqt_mode = True,
                                                        generate_figure = True,
                                                        output_format = 'excel'
                                                        )
```

These programmatic approach will return two pandas `DataFrame` objects:
    - `calculated_moment_results`: Contains the final moment magnitude results (averaged across all stations) for each successfully calculated earthquake ID.
    - `detailed_fitting_results`: Provides detailed spectral fitting results per station for each earthquake ID.

**2. CLI Approach**
```bash
    $ lqtmoment --wave-dir ..\tests\sample_tests_data\data\wave --cal-dir ..\tests\sample_tests_data\data\calibration --catalog-file ..\tests\sample_tests_data\results\lqt_catalog\lqt_catalog.xlsx --config-file ..\tests\sample_tests_data\calculation configuration\config_test.ini --fig-dir ..\tests\sample_tests_data\figures --output-dir ..\tests\sample_tests_data\results\calculation --id-start 1001 --id-end 1005 --create-figure --output-format excel
```

For more details please check out the full tutorials, which include `Tips` , `Notes` and `Cautions` for running the program effectively.

--------------
### References
This program relies on a robust scientific foundation, that refer to following resources:

* Abercrombie, R.E. (1995), "Earthquake Source Scaling Relationships from −1 to 5  M L  Using Seismograms Recorded at 2.5‐km Depth", Journal of Geophysical Research: Solid Earth, Vol.100, No.B12, hal. 24015–24036. http://doi.org/10.1029/95JB02397.
* Boore, D.M. dan Boatwright, J. (1984), "Average Body-Wave Radiation Coefficients", Bulletin of the Seismological Society of America, Vol.74, No.5, hal. 1615–1621. http://doi.org/10.1785/BSSA0740051615.
* Hanks, T.C. dan Kanamori, H. (1979), "A Moment Magnitude Scale", Journal of Geophysical Research: Solid Earth, Vol.84, No.B5, hal. 2348–2350. http://doi.org/10.1029/JB084iB05p02348.
* Havskov, J. dan Alguacil, G. (2016), Instrumentation in Earthquake Seismology, Springer International Publishing, Cham. http://doi.org/10.1007/978-3-319-21314-9.
* Havskov, J. dan Ottemoller, L. (2010), Routine Data Processing in Earthquake Seismology, Springer Netherlands, Dordrecht. http://doi.org/10.1007/978-90-481-8697-6.
* Maxwell, S. (2014), Microseismic Imaging of Hydraulic Fracturing: Improved Engineering of Unconventional Shale Reservoirs, Society of Exploration Geophysicists, US.

--------------
### Contributing
We kindly welcome contributions to **lqtmoment!** To get started, follow these steps:

- **Fork this Repository**: Clone the project to your GitHub account by forking it here https://github.com/bgjx/lqt-moment-magnitude.git
- **Clone Locally**: Download your forked repository to your machine using:
    ```bash
        git clone https://github.com/your-username/lqt-moment-magnitude.git 
    ```
- **Create a Branch**: Work on your changes in a new branch:
    ```bash
        git checkout -b your-feature-branch
    ```
- **Make Changes**: Implement your improvements or fixes, ensuring they align with the project's goals, or even new revolutionized ideas (you can contact the developer if you will by mail to edelo.arham@gmail.com)

- **Test Your Code**: You can extend the existed tests or add new ones if needed.
- **Commit & Push**: Save your work and push it to your fork:
    ```bash
        git add .
        git commit -m "Describe your changes here"
        git push origin your-feature-branch 
    ```
- **Submit and Pull Request**: Open the pull request (PR) on the original repository, detailing your changes and their purpose.

**Who can contributes ?**
* **Geophysicist/Seismologist**: To validate and improve moment magnitude calculation method for better accuracy.
* **Python Developer**: To optimize code or expand features.
* **Data Scientist**: To refine the ray tracing and spectral fitting algorithm to handle larger datasets effectively.
* **ML/AI Engineer**: To revolutionized the method to Artificial Intelligence approach.
* **Open-Source Enthusiast**: Anyone passionate about python programming and scientific research.

--------------
### Report Bugs
If you encounter any issues while running **lqtmoment**, please report them here: [Report Bugs](https://github.com/bgjx/lqt-moment-magnitude/issues). To help maintainer address the problem efficiently, include the following details:

- Your operating system and Python Version(e.g., Windows11 Python 3.11)
- A detailed description of the bug, including steps to reproduce it and any error messages.

Your contributions is crucial for improving the tool!