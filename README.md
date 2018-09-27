# Installation

This preconditioner experiment is now part of PETSc itself (as of version 3.10).  For native DMPlex + PetscDS discretisations, it is available with `-pc_type patch`.  The hookups to Firedrake (www.firedrakeproject.org) are now available there `-pc_type python -pc_python_type firedrake.PatchPC`.  Further development will happen in those respective projects.

## For posterity

This information is preserved for posterity.

Experiments using Firedrake and PETSc to precondition high order
discretisations of PDEs using the Schwarz decomposition methods of
Pavarino 1993 & 1994.

To build:
```
$ cd ssc
$ make
$ cd ..
$ pip install -e .
```
