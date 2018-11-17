from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

ext = [
   Extension(
       'occuspytial.icar.helpers.helper',
       sources=['occuspytial/icar/helpers/ctypes_helper.c']
   )
]

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='OccuSpytial',
    version='0.1',
    author='Zolisa Bleki',
    author_email='zolisa.bleki@gmail.com',
    description='A package for bayesian analysis of spatial occupancy models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD',
    url="https://github.com/zoj613/OccuSpytial",
    install_requires = [
        'loky', 'pandas','pypolyagamma',  # 'scikit-sparse' (optional)
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 2.7'
        'Programming Language :: Python :: 3.5'
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
