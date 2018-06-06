import numpy
cimport numpy
cimport petsc4py.PETSc as PETSc
from libc.stdint cimport uintptr_t

cdef extern from "petsc.h" nogil:
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    int PetscMalloc1(PetscInt, void*)
    int PetscFree(void*)

cdef extern from "petscpc.h" nogil:
    ctypedef enum PCPatchConstructType:
        PC_PATCH_STAR, PC_PATCH_VANKA, PC_PATCH_USER, PC_PATCH_PYTHON
    int PCPatchSetCellNumbering(PETSc.PetscPC, PETSc.PetscSection)
    int PCPatchSetDiscretisationInfo(PETSc.PetscPC, PetscInt, PETSc.PetscDM *,
                                     PetscInt *, PetscInt *,
                                     const PetscInt **,
                                     const PetscInt *,
                                     PetscInt,
                                     const PetscInt *,
                                     PetscInt,
                                     const PetscInt *)
    int PCPatchSetComputeOperator(PETSc.PetscPC, int (*)(PETSc.PetscPC, PetscInt, PETSc.PetscMat, PetscInt, const PetscInt *, PetscInt, const PetscInt *, void *) except -1, void*)
    int PCPatchSetConstructType(PETSc.PetscPC, PCPatchConstructType, int (*)(PETSc.PetscPC, PetscInt*, PETSc.PetscIS**, PETSc.PetscIS*, void *) except -1, void*)
    int PetscObjectReference(void *)

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    <void>SETERR(ierr)
    return -1

cdef inline object toInt(PetscInt value):
    return value
cdef inline PetscInt asInt(object value) except? -1:
    return value

cdef int PCPatch_ComputeOperator(
    PETSc.PetscPC pc,
    PetscInt point,
    PETSc.PetscMat mat,
    PetscInt ncell,
    const PetscInt *cells,
    PetscInt ndof,
    const PetscInt *dofmap,
    void *ctx) except -1 with gil:
    cdef PETSc.Mat Mat = PETSc.Mat()
    cdef PETSc.PC Pc = PETSc.PC()
    Pc.pc = pc
    Mat.mat = mat
    CHKERR( PetscObjectReference(<void*>mat) )
    CHKERR( PetscObjectReference(<void*>pc) )
    cdef object context = Pc.get_attr("__compute_operator__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    op(Pc, Mat, ncell, <uintptr_t>cells, <uintptr_t>dofmap, *args, **kargs)

cdef int PCPatch_UserPatchConstructionOperator(
    PETSc.PetscPC pc,
    PetscInt *nuserIS,
    PETSc.PetscIS **userIS,
    PETSc.PetscIS *userIterationSet,
    void *ctx) except -1 with gil:
    cdef PETSc.PC Pc = PETSc.PC()
    cdef PetscInt i
    Pc.pc = pc
    CHKERR( PetscObjectReference(<void*>pc) )
    cdef object context = Pc.get_attr("__patch_construction_operator__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    (patches, iterationSet) = op(Pc, *args, **kargs)

    nuserIS[0] = len(patches)
    CHKERR( PetscMalloc1(nuserIS[0], userIS) )

    for i from 0 <= i < nuserIS[0]:
        userIS[0][i] = (<PETSc.IS?>patches[i]).iset
        CHKERR( PetscObjectReference(<void*>userIS[0][i]) )

    userIterationSet[0] = (<PETSc.IS?>iterationSet).iset
    CHKERR( PetscObjectReference(<void*>userIterationSet[0]) )


cdef class PC(PETSc.PC):

    @staticmethod
    def cast(PETSc.PC input not None):
        cdef PC pc = PC()
        pc.pc = input.pc
        CHKERR( PetscObjectReference(<void *>pc.pc) )
        return pc

    def setPatchCellNumbering(self, PETSc.Section sec not None):
        CHKERR( PCPatchSetCellNumbering(self.pc, sec.sec) )

    def setPatchDiscretisationInfo(self, dms,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] bs,
                                   cellNodeMaps,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] subspaceOffsets,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] ghostBcNodes,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] globalBcNodes):
        cdef:
            PetscInt numSubSpaces
            PetscInt numGhostBcs, numGlobalBcs
            PetscInt *cnodesPerCell = NULL
            const PetscInt **ccellNodeMaps = NULL
            PETSc.PetscDM* cdms = NULL
            PetscInt i
            numpy.ndarray[PetscInt, ndim=2, mode="c"] tmp


        numSubSpaces = asInt(bs.shape[0])
        numGhostBcs = asInt(ghostBcNodes.shape[0])
        numGlobalBcs = asInt(globalBcNodes.shape[0])

        CHKERR( PetscMalloc1(numSubSpaces, &cnodesPerCell) )
        CHKERR( PetscMalloc1(numSubSpaces, &cdms) )
        CHKERR( PetscMalloc1(numSubSpaces, &ccellNodeMaps) )

        for i from 0 <= i < numSubSpaces:
            tmp = <numpy.ndarray?>(cellNodeMaps[i])
            ccellNodeMaps[i] = <PetscInt *> tmp.data
            cnodesPerCell[i] = cellNodeMaps[i].shape[1]
            cdms[i] = (<PETSc.DM?>dms[i]).dm

        CHKERR( PCPatchSetDiscretisationInfo(self.pc, numSubSpaces,
                                             cdms,
                                             <PetscInt *>bs.data,
                                             cnodesPerCell,
                                             ccellNodeMaps,
                                             <const PetscInt *>subspaceOffsets.data,
                                             numGhostBcs,
                                             <const PetscInt *>ghostBcNodes.data,
                                             numGlobalBcs,
                                             <const PetscInt *>globalBcNodes.data) )

        CHKERR ( PetscFree(ccellNodeMaps) )
        CHKERR ( PetscFree(cnodesPerCell) )
        CHKERR ( PetscFree(cdms) )

    def setPatchComputeOperator(self, operator, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__compute_operator__", context)
        CHKERR( PCPatchSetComputeOperator(self.pc, PCPatch_ComputeOperator, <void *>context) )

    def setPatchConstructType(self, typ, operator, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__patch_construction_operator__", context)
        CHKERR( PCPatchSetConstructType(self.pc, PC_PATCH_PYTHON, PCPatch_UserPatchConstructionOperator, <void *>context) )
