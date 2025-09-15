#!/bin/bash

# install local sources as package (for development)
pip uninstall unaiverse
pip install -e .

# install twine to publish package and create pip package (create only, see the "dist" folder)
pip install build twine
rm -rf dist/*
python -m build

# publish a test package and try to install it
# before doing it, create a file ~/.pypirc, and put the pypi token there, together with username = __token__, like this:
#
# [pypi]
#   username = __token__
#   password = <PUT_TOKEN_HERE>
#
# twine upload --verbose --repository testpypi dist/*
# pip uninstall unaiverse
# pip install --index-url https://test.pypi.org/simple/ --no-deps unaiverse

# publish on the final, public portal and try to install it
#twine upload dist/*
#pip uninstall unaiverse
#pip install unaiverse