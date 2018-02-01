from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
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

Set the environment variable PETSC_DIR / PETSC_ARCH to your local
PETSc base directory or install PETSc via pip.""")

if 'CC' not in env:
    env['CC'] = "mpicc"

petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]

link_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ssc"))

setup(name='ssc',
      cmdclass={'build_ext': build_ext},
      packages=["ssc"],
      ext_modules=[Extension('ssc.PatchPC',
                             sources=["ssc/PatchPC.pyx"],
                             include_dirs=include_dirs,
                             extra_link_args=["-L" + link_dir] +
                             ["-Wl,-rpath," + link_dir],
                             extra_compile_args=["-ggdb3", "-O0"],
                             libraries=["ssc"])])
