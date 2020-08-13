import importlib

import toml

from occuspytial import __version__


rel_version = importlib.import_module('docs.conf').release


def test_version():
    setup = toml.load('./pyproject.toml')
    assert __version__ == setup['tool']['poetry']['version']
    assert __version__ == rel_version
