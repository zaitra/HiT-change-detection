from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='HiT',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "terratorch",
        "torch",
        "torchvision",
        "torchgeo",
        "timm",
        "numpy",
        "strenum",
        ],
)

