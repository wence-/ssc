from distutils.core import setup
from distutils.extension import Extension
from glob import glob
from os import environ as env, path
import os
import sys
import numpy as np
import petsc4py


def get_petsc_dir():
    try:
        petsc_arch = env.get('PETSC_ARCH', '')
        petsc_dir = env['PETSC_DIR']
        if petsc_arch:
            return (petsc_dir, path.join(petsc_dir, petsc_arch))
        return (petsc_dir,)
    except KeyError:
        try:
            import petsc
            return (petsc.get_petsc_dir(), )
        except ImportError:
            sys.exit("""Error: Could not find PETSc library.

Set the environment variable PETSC_DIR to your local PETSc base
directory or install PETSc from PyPI as described in the manual:

http://firedrakeproject.org/obtaining_pyop2.html#petsc
""")

from Cython.Distutils import build_ext
import versioneer

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

if 'CC' not in env:
    env['CC'] = "mpicc"

petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]

setup(name='ssc',
      cmdclass=cmdclass,
      packages=["."],
      ext_modules=[Extension('PatchPC',
                             sources=["PatchPC.pyx"],
                             include_dirs=include_dirs,
                             extra_link_args = ["-L."] +
                             ["-Wl,-rpath,%s" % os.path.abspath(".")],
                             libraries=["ssc"])])
                             
