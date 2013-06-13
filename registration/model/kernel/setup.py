#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('kernel', parent_package, top_path)
    return config


if __name__ == '__main__':
    from distutils.core import setup
    setup(**configuration(top_path='').todict())
