Installation
============

.. note:: OccuSpytial requires python versions 3.6 or later.


Using ``pip``
-------------

The package can be installed using `pip <https://pip.pypa.io>`_.

.. code-block:: bash

    pip install occuspytial


From source
-----------

.. note:: Installing from source requires that `Cython <https://cython.readthedocs.io/en/latest/>`_ is available to successfully build the packages


Alternatively, it can be installed from source using `poetry <https://python-poetry.org>`_

.. code-block:: bash
    
    git clone https://github.com/zoj613/OccuSpytial.git
    cd OccuSpytial/
    poetry install


Testing
-------

.. note:: Make sure `pytest <https://docs.pytest.org/en/latest/>`_ is installed the environment before running the tests

After installation, the unit tests can be ran from the project's root directory
using

.. code-block:: bash
    
    pytest -v

