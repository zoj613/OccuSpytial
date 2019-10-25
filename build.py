from distutils.command.build_ext import build_ext
from distutils.core import Extension

ext = [
   Extension(
       'occuspytial.icar.helpers.helper',
       sources=['occuspytial/icar/helpers/ctypes_helper.c']
   )
]


# source: https://github.com/sdispater/pendulum/blob/1.x/build.py
def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': ext,
        'cmdclass': {'build_ext': build_ext}
    })
