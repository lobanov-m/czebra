import os
import setuptools


def parse_requirements():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, 'requirements.txt')) as f:
        lines = f.readlines()
    lines = [l for l in map(lambda l: l.strip(), lines) if l != '' and l[0] != '#']
    return lines


requirements = parse_requirements()


setuptools.setup(
    name='czebra',
    version='0.1',
    packages=setuptools.find_packages(),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)