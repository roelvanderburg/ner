#!/usr/bin/python3
import os
import re

from setuptools import find_packages
from setuptools import setup

NAME =               'ner-tag'
VERSION =            '0.1'
AUTHOR =             'coe cog'
AUTHOR_EMAIL =       'Dummy'
URL =                'https://devtools.belastingdienst.nl/bitbucket/projects/COOP/repos/ner-tag-roel'
DESCRIPTION =        'Package for automatic name-tagging via BSN identification numbers and classifying two-grams'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'Belastingdienst'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development']
PACKAGES = find_packages(exclude=("development",))

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        include_package_data = True,
        python_requires = '>=3',
        install_requires = ['certifi==2018.8.24',
                            'chardet==3.0.4',
                            'cycler==0.10.0',
                            'cymem==1.31.2',
                            'cytoolz==0.9.0.1',
                            'dill==0.2.8.2',
                            'ftfy==5.5.0',
                            'fuzzywuzzy==0.17.0',
                            'idna==2.7',
                            'kiwisolver==1.0.1',
                            'matplotlib==2.2.3',
                            'msgpack==0.5.6',
                            'msgpack-numpy==0.4.3.2',
                            'murmurhash==0.28.0',
                            'nl-core-news-sm==2.0.0',
                            'nltk==3.3',
                            'numpy==1.15.1',
                            'pandas==0.23.4',
                            'plac==0.9.6',
                            'preshed==1.0.1',
                            'pyparsing==2.2.1',
                            'python-dateutil==2.7.3',
                            'python-Levenshtein==0.12.0',
                            'pytz==2018.5',
                            'regex==2017.4.5',
                            'requests==2.19.1',
                            'scikit-learn==0.19.2',
                            'scipy==1.1.0',
                            'six==1.11.0',
                            'sklearn==0.0',
                            'spacy==2.0.12',
                            'thinc==6.10.3',
                            'toolz==0.9.0',
                            'tqdm==4.26.0',
                            'ujson==1.35',
                            'urllib3==1.26.5',
                            'wcwidth==0.1.7',
                            'wrapt==1.10.11'])
