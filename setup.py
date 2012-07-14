#! /usr/bin/env python
#
# Copyright (C) 2012 Michael Waskom <mwaskom@stanford.edu>

descr = """Moss: assorted utilities for neuroimaging and psychology"""

import os


DISTNAME = 'moss'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mwaskom/moss'
VERSION = '0.1.dev'

from numpy.distutils.core import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        packages=['moss', 'moss.tests'],
        scripts=["bin/" + s for s in ["check_mni_reg", "recon_movie",
                                      "recon_status", "recon_qc",
                                      "recon_process_stats", "ts_movie"]],
    )
