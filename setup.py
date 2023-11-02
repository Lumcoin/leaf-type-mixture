from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Leaf type mixture prediction with machine learning from satellite data.',
    author='Peter Hofinger',
    license='MIT',
    install_requires=[
        'ee',
        'eemont',
        'rasterio',
        'matplotlib',
        'numpy',
        'pandas',
        'pyproj',
        'requests',
        'scikit-learn',
        'skelm',
        'utm',
    ],
)
