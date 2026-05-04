from setuptools import setup, find_packages
setup(
    name="birdclef",
    version="1.0.0",
    description="Acoustic species identification — BirdCLEF+ 2026",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
