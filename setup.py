from setuptools import setup, find_packages

setup(
    name="network-traffic-classifier",
    version="1.0.0",
    description="Encrypted network traffic classification using flow-level features",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "scapy>=2.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "rich>=13.0.0",
    ],
)
