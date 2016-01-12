from setuptools import setup

setup(name='mmln',
      version='0.1',
      description='Machine learning tools for massively multi-labeled networks',
      url='https://github.com/stephenbach/mmln',
      author='Stephen Bach',
      author_email='bach@cs.stanford.edu',
      license='Apache Software License',
      packages=['mmln'],
      install_requires=[
            'networkx',
            'scikit-learn'
      ],
      zip_safe=False)
