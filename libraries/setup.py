import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learning-physical-properties-with-rnns-iamhectorotero",
    version="1.0.0",
    author="Hector Otero",
    author_email="7hector2@gmail.com",
    description="A package to train RNN in physical microworlds in a supervised or RL manner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamhectorotero/learning-physical-properties-with-rnns",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.6',
)
