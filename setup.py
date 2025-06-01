from setuptools import setup, find_packages

setup(
    name="pyclops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "mdtraj",
        "numpy>=2.0.0",
    ],
    extras_require={
        "visualization": [
            "matplotlib>=3.0.0",
            "nglview>=3.0.0",
            "MDAnalysis>=2.0.0",
        ],
        "structure": [
            "rosetta",  # Note: This might need specific installation instructions
            "openmm>=7.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ],
        "all": [
            "matplotlib>=3.0.0",
            "nglview>=3.0.0",
            "MDAnalysis>=2.0.0",
            "rosetta",  # Note: This might need specific installation instructions
            "openmm>=7.0.0",
            "scikit-learn>=1.0.0",
        ],
    },
    author="Ben du Pont",
    author_email="benfdup@gmail.com",
    description="Cyclic Loss Optimization for Peptide Structures", #CycLOPS, PycLOPS
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="peptide, structure, cyclization, computational-chemistry, loss-functions",
    url="https://github.com/benfdup/pyclops",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)