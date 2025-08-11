from setuptools import setup, find_packages

setup(
    name='rbl_pkg',
    version='0.1',
    description='Rank-Based Learning algorithm for binary classification ',
    author='Lulu Song',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
    ],
    python_requires='>=3.8',
)
