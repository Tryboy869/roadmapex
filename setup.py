"""
Setup configuration for Roadmapex
Exécution Prédictive Python by Anzize Daouda
"""

from setuptools import setup, find_packages
import os

# Lecture du README pour la description longue
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Lecture des requirements
with open(os.path.join(current_dir, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="roadmapex",
    version="0.1.0",
    author="Anzize Daouda",
    author_email="nexusstudio100@gmail.com",
    description="Exécution Prédictive Python - Transformez votre code réactif en orchestration stratégique",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tryboy869/roadmapex",
    project_urls={
        "Bug Reports": "https://github.com/Tryboy869/roadmapex/issues",
        "Source": "https://github.com/Tryboy869/roadmapex",
        "Documentation": "https://roadmapex.dev",
    },
    py_modules=["roadmapex"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0", 
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950"
        ],
        "benchmarks": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0"
        ]
    },
    keywords=[
        "python", "performance", "optimization", "execution", "predictive",
        "roadmap", "orchestration", "preloading", "intelligent", "automation",
        "gil", "interpreter", "speed", "efficiency"
    ],
    zip_safe=False,
    include_package_data=True,
)