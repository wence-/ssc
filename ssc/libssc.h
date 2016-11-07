#ifndef _PC_PATCH_H
#define _PC_PATCH_H
#include <petsc.h>
PETSC_EXTERN PetscErrorCode PCPatchInitializePackage(void);
PETSC_EXTERN PetscErrorCode PCCreate_PATCH(PC);
PETSC_EXTERN PetscErrorCode PCPatchSetDMPlex(PC, DM);
PETSC_EXTERN PetscErrorCode PCPatchSetDefaultSF(PC, PetscSF);
PETSC_EXTERN PetscErrorCode PCPatchSetCellNumbering(PC, PetscSection);
PETSC_EXTERN PetscErrorCode PCPatchSetDiscretisationInfo(PC, PetscSection,PetscInt,PetscInt,const PetscInt *,PetscInt,const PetscInt *);
PETSC_EXTERN PetscErrorCode PCPatchSetComputeOperator(PC, PetscErrorCode (*)(PC,Mat,PetscInt,const PetscInt *,PetscInt,const PetscInt *,void *),
                                                      void *);
#endif
