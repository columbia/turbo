from setuptools import setup, find_packages

setup(
    name="turbo-lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "termcolor",
        "redis",
        "torch==2.0.0",
        "loguru",
        "omegaconf",
        "numpy",
        "setuptools",
    ],
)
