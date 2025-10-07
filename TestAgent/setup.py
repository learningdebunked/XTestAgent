from setuptools import setup, find_packages

setup(
    name="testagentx",
    version="0.1.0",
    packages=find_packages(where="src") + ['evaluation'],
    package_dir={"": "src", "evaluation": "src/evaluation"},
    install_requires=[
        # Core Dependencies
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        # Additional dependencies from requirements.txt
        "gym>=0.26.2",
        "stable-baselines3>=2.1.0",
        "networkx>=3.2",
        "neo4j>=5.14.0",
        "javalang>=0.13.0",
        "tree-sitter>=0.20.4",
        "tree-sitter-java>=0.20.2",
        "pytest>=7.4.3",
        "coverage>=7.3.2",
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="TestAgentX: An advanced test generation and validation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/testagentx",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
