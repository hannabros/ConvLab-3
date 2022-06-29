'''
setup.py for ConvLab-3
'''
from setuptools import setup, find_packages

setup(
    name='convlab',
    version='3.0.0',
    packages=find_packages(),
    license='Apache',
    description='An Open-source Dialog System Platform',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    setup_requires=['setuptools-git'],
    install_requires=[
        'boto3',
        'matplotlib',
        'tabulate',
        'python-Levenshtein',
        'requests',
        'numpy',
        'nltk',
        'scipy',
        'tensorboard',
        'torch>=1.6',
        'transformers>=4.0',
        'datasets>=1.8',
        'seqeval',
        'spacy',
        'simplejson',
        'unidecode',
        'jieba',
        'embeddings',
        'visdom',
        'quadprog',
        'fuzzywuzzy',
        'json_lines',
        'gtts',
        'deepspeech',
        'pydub'
    ],
    extras_require={
        'develop': [
            "python-coveralls",
            "pytest-dependency",
            "pytest-mock",
            "requests-mock",
            "pytest",
            "pytest-cov",
            "checksumdir",
            "bs4",
            "lxml",
        ]
    },
    cmdclass={},
    entry_points={},
    include_package_data=True,
    url='https://github.com/ConvLab/ConvLab-3',
    author='convlab',
    author_email='convlab@googlegroups.com',
    python_requires='>=3.6',
    zip_safe=False
)
