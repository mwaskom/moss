#! /usr/bin/env python
#
# Copyright (C) 2012 Michael Waskom <mwaskom@stanford.edu>

descr = """Moss: statistical utilities for neuroimaging and cognitive science"""

import os


DISTNAME = 'moss'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
LICENSE = 'BSD (3-clause)'
URL = 'https://github.com/mwaskom/moss'
DOWNLOAD_URL = 'https://github.com/mwaskom/moss'
VERSION = '0.2.0'

from setuptools import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        URL=URL,
        download_url=DOWNLOAD_URL,
        packages=['moss', 'moss.tests'],
        scripts=["bin/" + s for s in ["check_mni_reg", "recon_movie",
                                      "recon_status", "recon_qc",
                                      "recon_process_stats", "ts_movie"]],
        classifiers=['Intended Audience :: Science/Research',
                     'Programming Language :: Python',
                     'License :: OSI Approved',
                     'Topic :: Scientific/Engineering',           
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        install_requires=["patsy", "pandas", "statsmodels", "scikit-learn", "six"],
    )
