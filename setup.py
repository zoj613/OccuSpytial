# Necessary hack for readthedocs to successfully build projects
# that use poetry with a C extensions. This "fake" setup.py file allows the
# project to be successfully installed inside the readthedoc environment.
from setuptools import setup
from build import *

global setup_kwargs

setup_kwargs = {}

build(setup_kwargs)
setup(**setup_kwargs)
