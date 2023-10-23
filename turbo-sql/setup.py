from setuptools import setup, find_packages

setup(
    name="turbo-sql",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sqlglot",
    ],
)
