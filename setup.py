import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

VERSION = "0.1.5"

setuptools.setup(
    name="n4ofunc",
    version=VERSION,
    author="noaione",
    author_email="noaione0809@gmail.com",
    description="N4O assorted VapourSynth functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["n4ofunc"],
    url="https://github.com/noaione/n4ofunc",
    package_data={
        "n4ofunc": ["py.typed"],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
