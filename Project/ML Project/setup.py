"""
setup.py — Project package setup for development install.
Run: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="hype_detection",
    version="1.0.0",
    description="Multimodal Fake Product Hype Detection",
    author="Research Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],          # managed via requirements.txt
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "isort", "flake8"],
    },
)
