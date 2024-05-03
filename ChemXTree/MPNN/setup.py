from setuptools import setup, find_packages


def read_requirements():
    return [
        'pandas==1.1.*',
        'chemprop==1.6.*',
        'rdkit-pypi==2022.9.*',
        'lightgbm==4.0.*',
        'xgboost==2.*'
    ]


setup(
    name='MPNN_Pipeline',
    version='0.0.1',
    packages=find_packages(), 
    install_requires=read_requirements() 
)
