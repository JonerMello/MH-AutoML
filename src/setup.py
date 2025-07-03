from setuptools import setup, find_packages

setup(
    name="src",
    version="0.4",
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
    python_requires=">=3.8",
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scipy==1.10.1',
        'tqdm==4.66.1',
        'joblib==1.3.2',
        'dask==2024.2.0',
        'psutil==5.9.8',
        'threadpoolctl==3.2.0',
        'scikit-learn==1.3.2',
        'lightgbm==4.3.0',
        'catboost==1.2.5',
        'optuna==3.6.1',
        'plotly==5.19.0',
        'spinners==0.0.24',
        'termcolor==2.4.0',
        'halo==0.0.31',
        'mlxtend==0.23.1',
        'tabulate==0.9.0',
        'imbalanced-learn==0.11.0',
        'shap==0.44.1',
        'lime==0.2.0.1',
        'seaborn==0.13.2',
        'mlflow==2.11.3',
        'timeout-decorator==0.5.0'
        ],
    extras_require={
        "pdf": [
            "weasyprint==60.2",
            "pdfkit",
            "playwright==1.44.0",
            "selenium==4.18.1",
            "Pillow==10.3.0",
            "reportlab==4.1.0"
        ]
    },
)

