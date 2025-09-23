#!/usr/bin/env python3
"""Setup script for ANPR Standalone package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "ANPR Standalone - License Plate Recognition System"

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements

setup(
    name="anpr-standalone",
    version="1.0.0",
    author="OsloVision",
    author_email="contact@oslovision.com",
    description="Standalone ANPR system with Streamlit UI and Norwegian registry integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/OsloVision/anpr-standalone",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "onnx": [
            "onnxruntime>=1.15.0",
        ],
        "tensorflow": [
            "tensorflow>=2.13.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anpr-ui=streamlit_ui:main",
            "anpr-batch=cli:batch_process",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
)