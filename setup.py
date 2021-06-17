from setuptools import setup


setup(
    name = "bigfeat",
    version = "0.1",
    #author = "",
    #author_email = "",
    description = ("Automated feature engineering library"),
    license = "MIT",
    keywords = "feature engineering", "machine learning", 
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
