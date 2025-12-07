from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="opengeodemand",
    version="0.2.1",  
    description="Open Source Geospatial Demand Simulation Engine",
    
    long_description=long_description,
    long_description_content_type='text/markdown',

    author="Dipit & Usman",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geopandas>=0.10",
        "pandas>=1.3",
        "numpy",
        "shapely",
        "osmnx>=1.9",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "folium",
        "mapclassify",
        "requests",
        "pyogrio"
    ],
    python_requires='>=3.8',
)