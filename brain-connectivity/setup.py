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
    # TODO
    # install_requires=[],
    # extras_require={
    #     "dev": ["isort", "black", "flake8"],
    # },
    project_urls={
        "Thesis repository": "https://github.com/janarez/brain-connectivity-thesis",
    },
)
