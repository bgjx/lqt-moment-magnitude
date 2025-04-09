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

* [Installation](#Installations)
* [Tutorials](#Tutorials)
* [Simple Example](#simple-Example)
* [References](#References)
* [Contributing](#Contributing)
* [Report Bugs](#Report-Bugs)

--------------
### Installation
The **lqtmoment** can be run in multiple platforms, including macOS, Windows and Linux in python 3.8 - 3.12. For installation, you can choose the following options:


build **lqtmoment** by cloning this repository using 
> python -m build
> python -m pip install dist\lqtmoment-0.1.1-py3-none-any.whl

or Via PyPI:

> python -m pip install --upgrade pip
> python -m pip install lqtmoment
