#!/bin/bash
set -e -u -x
# adapted from pypa's python-manylinux-demo and
# https://github.com/pypa/python-manylinux-demo/blob/7e24ad2c202b6f58c55654f89c141bba749ca5d7/travis/build-wheels.sh

# navigate to the root of the mounted project
cd $(dirname $0)

bin_arr=(
    /opt/python/cp36-cp36m/bin
    /opt/python/cp37-cp37m/bin
    /opt/python/cp38-cp38/bin
)

# add  python to image's path
export PATH=/opt/python/cp38-cp38/bin/:$PATH
# download && install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

function build_poetry_wheels
{
    # build wheels for 3.6-3.8 with poetry
    for BIN in "${bin_arr[@]}"; do
        rm -Rf build/*
        # install build requirement before building the wheel
        "${BIN}/python" ${HOME}/.poetry/bin/poetry run pip install numpy
        "${BIN}/python" ${HOME}/.poetry/bin/poetry build -f wheel
    done

    # add C libraries to wheels
    for whl in dist/*.whl; do
        auditwheel repair "$whl" --plat $1
        rm "$whl"
    done
}

build_poetry_wheels "$PLAT"
