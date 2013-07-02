#!/usr/bin/env python
from distutils.core import setup

DISTNAME = 'registration'
DESCRIPTION = 'registration framework'

LONG_DESCRIPTION = ''  # open('README.rst').read()
MAINTAINER = 'Demian Wassermann'
MAINTAINER_EMAIL = 'demian@bwh.harvard.edu'
URL = ''
LICENSE = ''
DOWNLOAD_URL = ''
VERSION = '0.1'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(quiet=True)
    config.add_subpackage('registration')
    return config


if __name__ == "__main__":
    requires = open('requirements.txt').readlines()
    for i, req in enumerate(requires):
        req = req.strip()
        if '>' in req:
            req = req.replace('>', '(>')
            req += ')'
        requires[i] = req

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        requires=requires,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            'scripts/register_tracts',
            'scripts/register_airways',
            'scripts/random_deform_airways',
            'scripts/deformation_to_nifti',
            'scripts/airway_math'
        ],
        **(configuration().todict())
    )
