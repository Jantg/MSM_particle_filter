import setuptools 

setuptools.setup(name = "MSM",
      version = "1.0",
      author = 'Jan Tomoya Greve, Jaeyeon Lee',
      author_email = 'jan.tomoya.greve@duke.edu, jaeyeon.lee@duke.edu',
      url = 'https://github.com/JJcomb/663project',
      py_modules = ['MSM'],
      packages = setuptools.find_packages(),
      scripts = ['run_sampledata.py'],
      python_requires = '>=3',
      license = 'MIT License',
      package_data={'':['*.csv']},
     )