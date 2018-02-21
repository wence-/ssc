#ifndef _PC_PATCH_H
#define _PC_PATCH_H
#include <petsc.h>
PETSC_EXTERN PetscErrorCode PCPatchInitializePackage(void);
PETSC_EXTERN PetscErrorCode PCCreate_PATCH(PC);
PETSC_EXTERN PetscErrorCode PCPatchSetCellNumbering(PC, PetscSection);
PETSC_EXTERN PetscErrorCode PCPatchSetDiscretisationInfo(PC, PetscInt, DM *, PetscInt *, PetscInt *, const PetscInt **, const PetscInt *, PetscInt, const PetscInt *, PetscInt, const PetscInt *);
PETSC_EXTERN PetscErrorCode PCPatchSetComputeOperator(PC, PetscErrorCode (*)(PC,Mat,PetscInt,const PetscInt *,PetscInt,const PetscInt *,void *),
                                                      void *);

typedef enum {PC_PATCH_STAR, PC_PATCH_VANKA, PC_PATCH_USER, PC_PATCH_PYTHON} PCPatchConstructType;

PETSC_EXTERN const char *const PCPatchConstructTypes[];
#endif
