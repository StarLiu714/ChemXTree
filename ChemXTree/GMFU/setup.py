from setuptools import setup, find_packages

setup(
    name='GMFU',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'omegaconf==2.3.*',
        'einops==0.6.*',
        'lightning_lite==1.8.6',
        'pytorch-lightning==1.8.*',
        'category_encoders==2.6.*',
    ],
)
