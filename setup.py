from setuptools import setup, find_packages

setup(
    name="pyclops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "MDAnalysis",
        "mdtraj",
    ],
    python_requires=">=3.10",
    author="Ben Dupont Jr",
    description="A Python package for protein structure analysis and optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
