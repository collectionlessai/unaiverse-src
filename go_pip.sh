# create pip package (create only, see the "dist" folder)
pip install build twine
python -m build

# clear artifacts
rm -rf unaiverse.egg-info

# install local sources as package (for development)
pip install -e .