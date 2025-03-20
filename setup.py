from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lqt-moment-magnitude",
    version="0.1.0",
    author= "Arham Zakki Edelo",
    author_email= "edelo.arham@gmail.com",
    description= "Rapid seismic moment magnitude calculation using advanced stochastic spectral fitting method in LQT components for very-local to teleseismic earthquakes",
    long_description= long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/bgjx/lqt-moment-magnitude",
    license="MIT",
    keywords='Seismology, Moment Magnitude, Spectral Fitting, LQT Component',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "scipy>=1.5.0",
        "scikit-optimize>=0.10.0"
        "obspy>=1.3.0",
        "tqdm>=4.60.0",
        "pathlib>=1.0.0"
        "configparser>=5.0.0",
    ],
    extras_require={
        "tests": ["pytest>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "LQT-Magnitude = lqt_moment_magnitude.main:main",
            "LQTCat-Build = lqt_moment_magnitude.catalog_builder:main"
        ]
    },
    python_requires = ">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic:: Scientific/Engineering :: Physics",
        "Topic:: Scientific/Engineering :: Geophysics",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta"
    ],
    include_package_data=True,
)