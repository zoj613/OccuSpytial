from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

ext = [
   Extension(
       'occuspytial.icar.helpers.helper',
       sources=['occuspytial/icar/helpers/ctypes_helper.c']
   )
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

descr = 'A package for bayesian analysis of spatial occupancy models'

setup(
    name='occuspytial',
    version='0.1',
    author='Zolisa Bleki',
    description=descr,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD',
    url="https://github.com/zoj613/OccuSpytial",
    install_requires=[
        'beautifultable==0.7.0',
        'pypolyagamma',
        'matplotlib>=3.1.0',
        'scipy==0.19.0'
    ],
    extras_require={
        'fast sparse matrix cholesky factorization': ['scikit-sparse']
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering'
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext
)
