import pathlib

from setuptools import find_packages, setup

root_path = pathlib.Path(__file__).parent.resolve()

setup(
    name="brain-connectivity",
    version="0.1.0",
    description="Source code for 'brain-connectivity' diploma thesis",
    long_description=(root_path / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/janarez/brain-connectivity",
    author="Jana Rezabkova",
    packages=find_packages(),
    python_requires=">=3.7.6, <4",
    install_requires=[
        "pandas==1.2.3",
        "matplotlib==3.4.1",
        "numpy==1.18.5",
        "scikit-learn==0.24.2",
        "seaborn==0.11.1",
        "statsmodels==0.13.1",
        "scipy==1.6.2",
        "tqdm==4.59.0",
        # "torch==1.10.2",
        # "torch-scatter==2.0.9",
        # "torch-sparse==0.6.13",
        # "torch-cluster==1.6.0",
        # "torch-geometric==2.0.4",
        "tensorboard==2.5.0",
        "torchinfo==1.6.1",
    ],
    extras_require={
        "dev": ["isort==5.9.3", "black==21.9b0", "flake8==4.0.1"],
    },
    project_urls={
        "Thesis repository": "https://github.com/janarez/brain-connectivity-thesis",
    },
)
