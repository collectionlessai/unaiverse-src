# create pip package (create only, see the "dist" folder)
pip install build twine
python -m build

# install local sources as package (for development)
pip install -e .