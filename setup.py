from setuptools import setup, find_packages

# read the long description from your README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="araras",
    version="1.0.0",
    description="A collection of tools for Machine Learning and Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matheus Ferreira Silva",
    author_email="matheusferreiravga@gmail.com",
    url="https://github.com/MatheusFS-dev/araras",
    license="GPL-3.0",
    license_files=["LICENSE.txt"],  # Use license_files instead of license-file
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow>=2.0.0",
        "optuna>=3.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "cycler>=0.11.0",
        "scikit-learn>=1.0.0",
        "psutil>=5.8.0",
        "scipy>=1.7.0",
        "fireducks-pandas>=0.9.0",
        "ipython>=7.0.0",
        "pretty_errors>=1.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
