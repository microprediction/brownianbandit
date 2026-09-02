import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="brownianbandit",
    version="0.1.0",
    description="Wavefront pruning of budgeted Brownian races",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/brownianbandit",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    packages=["brownianbandit"],
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={"test": ["pytest"], "demo": ["pandas", "matplotlib"]},
)
