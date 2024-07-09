from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    description="Tools AutoML Malwares android",
    author="Joner de Mello Assolin",
    author_email="jonermello@hotmail.com",
    #url="https://meu_site.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <3.10",
    install_requires=[
    'pandas',
    'numpy==1.22',
    'scipy',
    'tqdm',
    'joblib',
    'dask',
    'psutil',
    'threadpoolctl',
    'scikit-learn',
    'lightgbm',
    'catboost',
    'optuna',
    'plotly',
    'spinners',
    'termcolor',
    'halo',
    'mlxtend',
    'tabulate',
    'imbalanced-learn',
    'shap',
    'lime',
    'seaborn',
    'mlflow==2.11.3',
    'timeout-decorator'
        ],
)

