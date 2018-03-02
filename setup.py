from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from os import environ as env, path
import sys
import numpy as np
import petsc4py
import subprocess


""" MPI stuff """
try:
    ex_flags = subprocess.run(['mpicc', '-show'], stdout=subprocess.PIPE)
    ex_flags = ex_flags.stdout.decode('utf-8')
    ex_flags = ex_flags.strip()
    ex_flags = ex_flags.split()[1:]
    print(ex_flags)
    print(type(ex_flags[0]))
except IOError:
    print('Cannot query mpicxx compiler')
    print('Make sure that mpicxx exists, and returns flags with "mpicc -show"')
    sys.exit(1)

ex_libs = [x[2:] for x in ex_flags if x[:2] == '-l']

if 'CC' not in env:
    env['CC'] = "mpicc"


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


petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]
include_dirs += ["ssc/"]
link_dirs = ["%s/lib" % d for d in petsc_dirs]

setup(name='ssc',
      cmdclass={'build_ext': build_ext},
      packages=["ssc"],
      ext_modules=[Extension('ssc.PatchPC',
                             sources=["ssc/PatchPC.pyx", "ssc/libssc.c"],
                             include_dirs=include_dirs,
                             library_dirs=link_dirs,
                             runtime_library_dirs=link_dirs,
                             libraries=ex_libs + ["petsc"],
                             extra_compile_args=ex_flags)])
