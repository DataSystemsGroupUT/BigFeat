from setuptools import setup


setup(
    name = "bigfeat",
    version = "0.1",
    #author = "Shota Amashukeli, Hassan Eldeeb, Radwa Elshawi",
    #author_email = "firstName.lastName@ut.ee",
    description = ("Automated feature engineering library"),
    license = "MIT",
    keywords = "feature engineering", "machine learning", "automl", "feature extraction", "feature selection",
    url = "https://github.com/DataSystemsGroupUT/BigFeat",
    packages=['bigfeat'],
    long_description=("Automated feature engineering library"),
          install_requires=[
          'pandas',
          'numpy',
          'scikit-learn',
          'lightgbm',
      ],
)
