from setuptools import setup

setup(
    description = ("Automated feature engineering library"),
    name = "bigfeat",
    version = "0.1",
    #author = "Hassan Eldeeb, Shota Amashukeli, Radwa Elshawi",
    #author_email = "firstName.lastName@ut.ee",
    license = "MIT",
    keywords = ["feature engineering", "machine learning", "automl", "feature extraction", "feature selection"],
    url = "https://github.com/DataSystemsGroupUT/BigFeat",
    packages=['bigfeat'],
    long_description=("Automated feature engineering library"),
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'lightgbm',
        'scipy',
        'psutil',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
