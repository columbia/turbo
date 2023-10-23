from setuptools import setup, find_packages

setup(
    name="dummmy-dp-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandasql",
        "numpy",
        "pandas",
        "SQLAlchemy==1.4.46",
    ],
)
