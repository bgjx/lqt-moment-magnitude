[![Follow me on Twitter](https://img.shields.io/badge/Follow-@zakki_edelo-blue?logo=x&logoColor=white&style=flat)](https://twitter.com/zakki_edelo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arham_zakki-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/arhamzakki)
[![CI - Tests](https://github.com/bgjx/lqt-moment-magnitude/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/bgjx/lqt-moment-magnitude/actions/workflows/ci-cd.yml)
[![GitHub Issues](https://img.shields.io/github/issues/bgjx/lqt-moment-magnitude?style=flat)](https://github.com/bgjx/lqt-moment-magnitude/issues)
[![GitHub Commits](https://img.shields.io/github/last-commit/bgjx/lqt-moment-magnitude?style=flat)](https://github.com/bgjx/lqt-moment-magnitude/commits/main/)
[![PyPI](https://img.shields.io/pypi/v/lqtmoment?style=flat$logo=pypi)](https://pypi.org/project/lqtmoment/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)


#### What is it?

**lqtmoment** is python package designed for moment magnitude calculations using pure P, SV, and SH components in the LQT ray coordinate system. It leverages rapid ray tracing to compute incidence angles for component rotation and employs fast spectral fitting to find optimal solution, ensuring high accuracy and efficient automated computation.

The **lqtmoment** includes modules for building input catalog format, performing moment magnitude calculation, visualizations, and simple data analysis.

Contact the developer: Arham Zakki Edelo (edelo.arham@gmail.com)

* [Installation](###Installations)
* [Tutorials](#Tutorials)
* [Simple Example](#simple-Example)
* [References](#References)
* [Contributing](#Contributing)
* [Report Bugs](#Report-Bugs)

--------------
### Installations
**lqtmoment** can be installed and run on multiple platforms, including macOS, Windows, and Linux, using Python versions 3.8 to 3.12. Choose one of the following installation methods:

#### Option 1: Via Anaconda
If you have Anaconda installed, create and activate a new environment (for clean installation), then install `lqtmoment` from the `bgjx` channel:

```bash
    conda create -n lqtmoment python=3.8
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