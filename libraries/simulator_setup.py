import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simulator-iamhectorotero",
    version="0.0.1",
    author="Hector Otero",
    author_email="7hector2@gmail.com",
    description="A simulator for physical microworlds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamhectorotero/diss",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.6',
)
