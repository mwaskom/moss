Assorted Scientific Utilities
=============================

Moss is a library of functions, classes, and scripts to that may be useful
for analyzing scientific data. Because this package is developed for
neuroimaging and cognitive science, there is probably some bias towards
applications that are useful in that domain. However, the functions are
intended to be written in as general and lightweight a fashion as possible.


Dependencies
------------

- Python 2.7

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](matplotlib.sourceforge.net)

- [pandas](http://pandas.pydata.org/)

- [scikit-learn](http://scikit-learn.org/stable/)

- [six](http://pythonhosted.org/six/)



Installation
------------

To install the released version, just do

    pip install -U moss

However, I update the code pretty frequently, so you may want to clone the
github repository and install with

    python setup.py install


Testing
-------

[![Build Status](https://travis-ci.org/mwaskom/moss.png?branch=master)](https://travis-ci.org/mwaskom/moss)

You can exercise the full test suite by running `nosetests` in the source
directory.

Note that some of the statistical tests depend on randomly generated data and
fail from time to time because of this.

License
-------

Moss is released under a BSD (3-clause) license.


Development
-----------

http://github.com/mwaskom/moss

Please report any bugs you encounter on the Github issue tracker.
