from setuptools import setup, find_packages
from pathlib import Path

def load_requirements(filename):
    with open(filename) as f:
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name='gecko',
    version='0.0.1',
    description='',
    author='Farros Alferro',
    author_email='farros.alferro.t3@dc.tohoku.ac.jp',
    packages=find_packages(),
    url='https://github.com/farrosalferro?tab=repositories',
    install_requires=load_requirements("requirements.txt")
)