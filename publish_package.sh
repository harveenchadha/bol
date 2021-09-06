rm -rf bol_libary.egg-info
rm -rf build
rm -rf dist

python setup.py sdist bdist_wheel
python -m twine upload dist/*