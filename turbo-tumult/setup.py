from setuptools import setup, find_packages

setup(
    name="turbo-tumult",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sqlglot",
        "tmlt-analytics==0.7.3",
        "tmlt-core",
],
)
