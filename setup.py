from setuptools import setup, find_packages

setup(
    name="MALIBOO",
    version="1.0.0",
    url="https://github.com/brunoguindani/MALIBOO",
    packages=find_packages(),
    author="Bruno Guindani",
    author_email="bruno.guindani@polimi.it",
    description="Machine Learning in Bayesian Optimization",
    long_description="Bayesian Optimization algorithms integrated with Machine Learning techniques",
    install_requires=[
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "setuptools"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ]
)
