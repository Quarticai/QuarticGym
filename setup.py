import setuptools
from quarticgym._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setuptools.setup(
    name="quarticgym", 
    version=__version__,
    author="Quartic",
    author_email="mohan@quartic.ai",
    description="Process Control environments using gym API protocols",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quarticai/QuarticGym.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
