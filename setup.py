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
VERSION = '0.4.2'

from setuptools import setup

def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    needed_deps = ["IPython", "numpy", "scipy", "matplotlib",
                   "sklearn", "pandas", "statsmodels", "patsy",
                   "six", "seaborn"]
    missing_deps = []
    for dep in needed_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        missing = (", ".join(missing_deps)
                   .replace("sklearn", "scikit-learn")
                   .replace("skimage", "scikit-image"))
        raise ImportError("Missing dependencies: %s" % missing)


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    import sys
    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands',
                            '--version',
                            'egg_info',
                            'clean'))):
        check_dependencies()

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        URL=URL,
        download_url=DOWNLOAD_URL,
        packages=['moss', 'moss.tests',
                  'moss.nipy', 'moss.nipy.volumes', 'moss.nipy.transforms',
                  'moss.external'],
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
    )
