import numpy
cimport numpy
cimport petsc4py.PETSc as PETSc
from libc.stdint cimport uintptr_t

cdef extern from "petsc.h" nogil:
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE

cdef extern from "libssc.h" nogil:
    int PCPatchSetDMPlex(PETSc.PetscPC, PETSc.PetscDM)
    int PCPatchSetDefaultSF(PETSc.PetscPC, PETSc.PetscSF)
    int PCPatchSetCellNumbering(PETSc.PetscPC, PETSc.PetscSection)
    int PCPatchSetDiscretisationInfo(PETSc.PetscPC, PETSc.PetscSection,
                                     PetscInt, PetscInt,
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


cdef class PC(petsc4py.PETSc.PC):

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

    def setPatchDiscretisationInfo(self, PETSc.Section dofSection,
                                   bs, numpy.ndarray[PetscInt, ndim=2, mode="c"] cellNodeMap,
                                   numpy.ndarray[PetscInt, ndim=1, mode="c"] bcNodes):
        cdef:
            PetscInt cbs = asInt(bs)
            PetscInt nodesPerCell = cellNodeMap.shape[1]
            PetscInt numBcs = bcNodes.shape[0]

        CHKERR( PCPatchSetDiscretisationInfo(self.pc, dofSection.sec, cbs,
                                             nodesPerCell,
                                             <const PetscInt *>cellNodeMap.data,
                                             numBcs,
                                             <const PetscInt *>bcNodes.data) )

    def setPatchComputeOperator(self, operator, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__compute_operator__", context)
        CHKERR( PCPatchSetComputeOperator(self.pc, PCPatch_ComputeOperator, <void *>context) )


PCPatchInitializePackage()
