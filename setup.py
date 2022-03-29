import setuptools

from n4ofunc import __author__, __title__, __version__

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()


setuptools.setup(
    name=__title__,
    version=__version__,
    author=__author__,
    author_email="noaione0809@gmail.com",
    description="n4ofunc assorted VapourSynth functions",
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
