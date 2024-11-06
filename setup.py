import os

from setuptools import setup, find_packages


def read(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="scann-model",
    version="1.0",
    author="Vu Tien-Sinh",
    author_email="sinh.vt@jaist.ac.jp",
    url="https://github.com/sinhvt3421/scann--material",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.8.0,<=2.12.0",
        "scikit-learn",
        "numpy",
        "h5py",
        "pyyaml",
        "ase",
        "pymatgen>=2023.8.10",
    ],
    include_package_data=True,
    keywords=["materials", "science", "interpretable", "deep", "attention", "networks", "neural"],
    extras_require={"test": ["pytest", "pytest-datadir", "pytest-benchmark"]},
    license="MIT",
    description="SCANN - Self-Consistent Atention-based Neural Network",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
