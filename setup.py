from setuptools import setup, find_packages

setup(
    name="analognas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "networkx>=2.6.0",
        "pyyaml>=5.4.1",
    ],
    python_requires=">=3.8",
)
