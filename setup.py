from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LQTMomentMag",
    version="1.0.0",
    author= "Arham Zakki Edelo",
    author_email= "edelo.arham@gmail.com",
    description= "Calculate seismic moment magnitude in full LQT energy components",
    long_description= long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/bgjx/LQTMomentMag",
    license="MIT",
    keywords='Seismology, Moment Magnitude, Spectral Fitting, LQT Component',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "scipy>=1.5.0",
        "obspy>=1.3.0",
        "tqdm>=4.60.0",
        "configparser>=5.0.0",
    ],
    tests_requires = ["pytest>=7.0.0"],
    entry_points={
        "console_scripts": [
            "LQTMwCalc = LQTMomentMag.main:main",
        ]
    },
    python_requires = ">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic:: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta"
    ],
    include_package_data=True,
)