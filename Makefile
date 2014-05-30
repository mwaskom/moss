export SHELL := /bin/bash

coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package moss

lint:

	pyflakes -x W moss
	pep8 --exclude leastsqbound.py,nipy moss
