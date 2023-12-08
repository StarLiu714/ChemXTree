from setuptools import setup, find_packages


def read_requirements():
    return [
        'torch==1.8.1',
        'category-encoders==2.5.*',
        'xgboost==2.0.*',
        'chemprop==1.6.*',
        'numpy==1.17.2',
        'pandas==1.1.5',
        'scikit-learn==1.0.0',
        'pytorch-lightning==1.8.*',
        'omegaconf==2.0.1',
        'torchmetrics==0.11.*',
        'tensorboard==2.2.0,!=2.5.0',
        'protobuf==3.20.*',
        'pytorch-tabnet==4.0.*',
        'PyYAML==5.4.1,!=5.1.*',
        # 'importlib-metadata==1.0.*,!=0.12.*', # Uncomment this line if needed
        'matplotlib==3.1.*',
        'ipywidgets',
        "dataclasses; python_version == '3.6'",
        'einops==0.6.*',
        'rich==10.2.2',
        'rdkit-pypi==2022.9.*',
        'lightgbm==4.0.*'
    ]




setup(
    name='ChemXTree',
    version='0.0.1',
    packages=find_packages(), 
    install_requires=read_requirements() 
)
