import numpy
cimport numpy
cimport petsc4py.PETSc as PETSc
from libc.stdint cimport uintptr_t

cdef extern from "petsc.h" nogil:
    int PetscMalloc1(PetscInt, void*)
    int PetscFree(void*)
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE

cdef extern from "libssc.h" nogil:
    int PCPatchSetDMPlex(PETSc.PetscPC, PETSc.PetscDM)
    int PCPatchSetDefaultSF(PETSc.PetscPC, PETSc.PetscSF)
    int PCPatchSetCellNumbering(PETSc.PetscPC, PETSc.PetscSection)
    int PCPatchSetDiscretisationInfo(PETSc.PetscPC, PetscInt, PETSc.PetscSection *,
                                     PetscInt *, PetscInt *,
                                     const PetscInt **,
                                     const PetscInt *,
                                     PetscInt,
                                     const PetscInt *)
    int PCPatchSetComputeOperator(PETSc.PetscPC, int (*)(PETSc.PetscPC, PETSc.PetscMat, PetscInt, const PetscInt *, PetscInt, const PetscInt *, void *) except -1, void*)
    int PCCreate_PATCH(PETSc.PetscPC)
    int PetscObjectReference(void *)
    int PCPatchInitializePackage()

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


cdef class PC(PETSc.PC):

    @staticmethod
    def cast(PETSc.PC input not None):
        cdef PC pc = PC()
        pc.pc = input.pc
        CHKERR( PetscObjectReference(<void *>pc.pc) )
        return pc

    def setPatchDMPlex(self, PETSc.DM dm not None):
        CHKERR( PCPatchSetDMPlex(self.pc, dm.dm) )

    def setPatchDefaultSF(self, PETSc.SF sf not None):
        CHKERR( PCPatchSetDefaultSF(self.pc, sf.sf) )

    def setPatchCellNumbering(self, PETSc.Section sec not None):
        CHKERR( PCPatchSetCellNumbering(self.pc, sec.sec) )

    def setPatchDiscretisationInfo(self, dofSections,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] bs,
                                   cellNodeMaps,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] subspaceOffsets,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] bcNodes):
        cdef:
            PetscInt numSubSpaces = bs.shape[0]
            PetscInt numBcs = bcNodes.shape[0]
            PetscInt *cnodesPerCell = NULL
            const PetscInt **ccellNodeMaps = NULL
            PETSc.PetscSection* csections = NULL
            PetscInt i
            numpy.ndarray[PetscInt, ndim=2, mode="c"] tmp


        CHKERR( PetscMalloc1(numSubSpaces, &ccellNodeMaps) )
        CHKERR( PetscMalloc1(numSubSpaces, &cnodesPerCell) )
        CHKERR( PetscMalloc1(numSubSpaces, &csections) )

        for i from 0 <= i < numSubSpaces:
            tmp = <numpy.ndarray?>(cellNodeMaps[i])
            ccellNodeMaps[i] = <PetscInt *> tmp.data
            cnodesPerCell[i] = cellNodeMaps[i].shape[1]
            csections[i] = <PETSc.PetscSection> dofSections[i].sec

        CHKERR( PCPatchSetDiscretisationInfo(self.pc, numSubSpaces,
                                             csections,
                                             <PetscInt *>bs.data,
                                             cnodesPerCell,
                                             ccellNodeMaps,
                                             <const PetscInt *>subspaceOffsets.data,
                                             numBcs,
                                             <const PetscInt *>bcNodes.data) )

        CHKERR ( PetscFree(ccellNodeMaps) )
        CHKERR ( PetscFree(cnodesPerCell) )
        CHKERR ( PetscFree(csections) )

    def setPatchComputeOperator(self, operator, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__compute_operator__", context)
        CHKERR( PCPatchSetComputeOperator(self.pc, PCPatch_ComputeOperator, <void *>context) )


PCPatchInitializePackage()
