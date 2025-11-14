from setuptools import setup, find_packages

setup(
    name="opengeodemand",          
    version="0.1.0",            
    packages=find_packages(),  
    install_requires=[],        
    author="Anupam Sobti, Dipit Golecha, Usman Akinyemi",
    author_email="usmanakinyemi202@gmail.com",
    description="opengeodemand",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    licence="",
    url="https://github.com/Geospatial-Computer-Vision-Group/opengeodemand",
    classifiers=[
    ],
    keywords=[],
    include_package_data=True,
)

