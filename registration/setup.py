#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import setup, Extension
    from numpy.distutils.system_info import lapack_info, lapack_mkl_info, lapack_opt_info
    from os.path import join, exists

    config = Configuration('registration', parent_package, top_path)
    lapack_libs = []
    lapack_lib_dirs = []
    lapack_include_dirs = []
    lapack_extra_compile_args = []
    lapack_extra_link_args = []
    for l in [lapack_mkl_info().get_info(), lapack_opt_info().get_info(), lapack_info().get_info()]:
        try:
            lapack_libs += l['libraries']
            lapack_lib_dirs += l['library_dirs']
            lapack_include_dirs += l['include_dirs']
            lapack_extra_compile_args += l['extra_compile_args']
            lapack_extra_link_args += l['extra_link_args']
            break
        except:
            pass
    config.add_subpackage('metric')
    config.add_subpackage('model')

    return config


if __name__ == '__main__':
    from distutils.core import setup
    setup(**configuration(top_path='').todict())
