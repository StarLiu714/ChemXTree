from setuptools import setup, find_packages


def read_requirements():
    return [
        'torch>=1.8.*',
        'category-encoders==2.5.*',
        'xgboost==2.*',
        'chemprop==1.6.*',
        'numpy>=1.17.*',
        'pandas<=2.1.*',
        'scikit-learn==1.*',
        'pytorch-lightning==1.8.*',
        'omegaconf==2.0.1',
        'torchmetrics==0.11.*',
        'tensorboard>=2.2.0,!=2.5.0',
        'protobuf==3.20.*',
        'pytorch-tabnet==4.0.*',
        'PyYAML>=5.2',
        # 'importlib-metadata==1.0.*,!=0.12.*', # Uncomment this line if needed
        'matplotlib==3.*',
        'ipywidgets',
        "dataclasses",
        'einops==0.6.*',
        'rich==10.2.*',
        'rdkit==2023.*',
        'lightgbm==4.*'
    ]




setup(
    name='ChemXTree',
    version='0.0.1',
    description=(
        "A graph-based drug discovery package for molecular prediction"),
    author="Star Xinxin Liu",
    author_email="StarLiu@seas.upenn.edu",
    license="MIT License",
    url="https://github.com/StarLiu714/ChemXTree/",
    packages=find_packages(), 
    install_requires=read_requirements(),
    python_requires=">=3.6"

)
