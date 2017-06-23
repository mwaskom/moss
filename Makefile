export SHELL := /bin/bash

test:

	py.test

coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package moss

lint:

	pyflakes -x W moss
	pep8 --exclude external moss
