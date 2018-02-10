#include <petsc/private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petsc.h>
#include <petsc/private/hash.h>
#include <petscsf.h>
#include <libssc.h>

PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Scatter, PC_Patch_Apply;

static PetscBool PCPatchPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PCPatchInitializePackage"
PETSC_EXTERN PetscErrorCode PCPatchInitializePackage(void)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;

    if (PCPatchPackageInitialized) PetscFunctionReturn(0);
    PCPatchPackageInitialized = PETSC_TRUE;
    ierr = PCRegister("patch", PCCreate_PATCH); CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCPATCHCreate", PC_CLASSID, &PC_Patch_CreatePatches); CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCPATCHComputeOp", PC_CLASSID, &PC_Patch_ComputeOp); CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCPATCHSolve", PC_CLASSID, &PC_Patch_Solve); CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCPATCHApply", PC_CLASSID, &PC_Patch_Apply); CHKERRQ(ierr);
    ierr = PetscLogEventRegister("PCPATCHScatter", PC_CLASSID, &PC_Patch_Scatter); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

typedef struct {
    DM              dm;         /* DMPlex object describing mesh
                                 * topology (need not be the same as
                                 * PC's DM) */
    PetscSF          defaultSF;
    PetscSection    *dofSection;
    PetscSection     cellCounts;
    PetscSection     cellNumbering; /* Numbering of cells in DM */
    PetscSection     gtolCounts;   /* Indices to extract from local to
                                   * patch vectors */
    PetscInt         nsubspaces;   /* for mixed problems */
    PetscInt        *subspaceOffsets; /* offsets for calculating concatenated numbering for mixed spaces */
    PetscSection     bcCounts;
    IS               cells;
    IS               dofs;
    IS               bcNodes;
    IS               gtol;
    IS              *bcs;

    PetscSection     multBcCounts; /* these are the corresponding BC objects for just the actual */
    IS              *multBcs;      /* Only used for multiplicative smoothing to recalculate residual */

    MPI_Datatype     data_type;
    PetscBool        free_type;

    PetscBool        save_operators; /* Save all operators (or create/destroy one at a time?) */
    PetscBool        partition_of_unity; /* Weight updates by dof multiplicity? */
    PetscBool        multiplicative; /* Gauss-Seidel or Jacobi? */
    PetscInt         npatch;     /* Number of patches */
    PetscInt        *bs;            /* block size (can come from global
                                    * operators?) */
    PetscInt        *nodesPerCell;
    PetscInt         totalDofsPerCell;
    const PetscInt **cellNodeMap; /* Map from cells to nodes */

    KSP             *ksp;        /* Solvers for each patch */
    Vec              localX, localY;
    Vec              dof_weights; /* In how many patches does each dof lie? */
    Vec             *patchX, *patchY; /* Work vectors for patches */
    Mat             *mat;        /* Operators */
    Mat             *multMat;        /* Operators for multiplicative residual calculation */
    MatType          sub_mat_type;
    PetscErrorCode  (*usercomputeop)(PC, Mat, PetscInt, const PetscInt *, PetscInt, const PetscInt *, void *);
    void            *usercomputectx;

    PetscErrorCode  (*patchconstructop)(void*, DM, PetscInt, PetscHashI); /* patch construction */
    PetscInt         codim; /* dimension or codimension of entities to loop over; */
    PetscInt         dim;   /* only oxne of them can be set */
    PetscInt         vankaspace; /* What's the constraint space, when we're doing Vanka */
    PetscInt         vankadim;   /* In Vanka construction, should we eliminate any entities of a certain dimension? */
} PC_PATCH;

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetDMPlex"
PETSC_EXTERN PetscErrorCode PCPatchSetDMPlex(PC pc, DM dm)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->dm = dm;
    ierr = PetscObjectReference((PetscObject)dm); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetSaveOperators"
PETSC_EXTERN PetscErrorCode PCPatchSetSaveOperators(PC pc, PetscBool flg)
{
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->save_operators = flg;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetPartitionOfUnity"
PETSC_EXTERN PetscErrorCode PCPatchSetPartitionOfUnity(PC pc, PetscBool flg)
{
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->partition_of_unity = flg;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetMultiplicative"
PETSC_EXTERN PetscErrorCode PCPatchSetMultiplicative(PC pc, PetscBool flg)
{
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->multiplicative = flg;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetDefaultSF"
PETSC_EXTERN PetscErrorCode PCPatchSetDefaultSF(PC pc, PetscSF sf)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->defaultSF = sf;
    ierr = PetscObjectReference((PetscObject)sf); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetCellNumbering"
PETSC_EXTERN PetscErrorCode PCPatchSetCellNumbering(PC pc, PetscSection cellNumbering)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    patch->cellNumbering = cellNumbering;
    ierr = PetscObjectReference((PetscObject)cellNumbering); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCPatchSetDiscretisationInfo"
PETSC_EXTERN PetscErrorCode PCPatchSetDiscretisationInfo(PC pc, PetscInt nsubspaces,
                                                         PetscSection *dofSection,
                                                         PetscInt *bs,
                                                         PetscInt *nodesPerCell,
                                                         const PetscInt **cellNodeMap,
                                                         const PetscInt *subspaceOffsets,
                                                         PetscInt numBcs,
                                                         const PetscInt *bcNodes)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;

    ierr = PetscMalloc1(nsubspaces, &patch->dofSection); CHKERRQ(ierr);
    ierr = PetscMalloc1(nsubspaces, &patch->bs); CHKERRQ(ierr);
    ierr = PetscMalloc1(nsubspaces, &patch->nodesPerCell); CHKERRQ(ierr);
    ierr = PetscMalloc1(nsubspaces, &patch->cellNodeMap); CHKERRQ(ierr);
    ierr = PetscMalloc1(nsubspaces+1, &patch->subspaceOffsets); CHKERRQ(ierr);

    patch->nsubspaces = nsubspaces;
    patch->totalDofsPerCell = 0;
    for (int i = 0; i < nsubspaces; i++) {
        patch->dofSection[i] = dofSection[i];
        ierr = PetscObjectReference((PetscObject)dofSection[i]); CHKERRQ(ierr);
        patch->bs[i] = bs[i];
        patch->nodesPerCell[i] = nodesPerCell[i];
        patch->totalDofsPerCell += nodesPerCell[i]*bs[i];
        patch->cellNodeMap[i] = cellNodeMap[i];
        patch->subspaceOffsets[i] = subspaceOffsets[i];
    }
    patch->subspaceOffsets[nsubspaces] = subspaceOffsets[nsubspaces];
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numBcs, bcNodes, PETSC_COPY_VALUES, &patch->bcNodes); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetSubMatType"
PETSC_EXTERN PetscErrorCode PCPatchSetSubMatType(PC pc, MatType sub_mat_type)
{
    PetscErrorCode ierr;
    PC_PATCH      *patch = (PC_PATCH *)pc->data;
    PetscFunctionBegin;
    if (patch->sub_mat_type) {
        ierr = PetscFree(patch->sub_mat_type); CHKERRQ(ierr);
    }
    ierr = PetscStrallocpy(sub_mat_type, (char **)&patch->sub_mat_type); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchSetComputeOperator"
PETSC_EXTERN PetscErrorCode PCPatchSetComputeOperator(PC pc, PetscErrorCode (*func)(PC, Mat, PetscInt,
                                                                                    const PetscInt *,
                                                                                    PetscInt,
                                                                                    const PetscInt *,
                                                                                    void *),
                                                      void *ctx)
{
    PC_PATCH *patch = (PC_PATCH *)pc->data;

    PetscFunctionBegin;
    /* User op can assume matrix is zeroed */
    patch->usercomputeop = func;
    patch->usercomputectx = ctx;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchCompleteCellPatch"
/* On entry, ht contains the topological entities whose dofs we are responsible for solving for;
   on exit, cht contains all the topological entities we need to compute their residuals.
   In full generality this should incorporate knowledge of the sparsity pattern of the matrix;
   here we assume a standard FE sparsity pattern.*/
static PetscErrorCode PCPatchCompleteCellPatch(DM dm, PetscHashI ht, PetscHashI cht)
{
    PetscErrorCode    ierr;
    PetscHashIIter    hi;
    PetscInt          entity;

    PetscFunctionBegin;


    PetscHashIClear(cht);
    PetscHashIIterBegin(ht, hi);
    while (!PetscHashIIterAtEnd(ht, hi)) {
        PetscInt       starSize, closureSize;
        PetscInt      *star = NULL, *closure = NULL;

        PetscHashIIterGetKey(ht, hi, entity);
        PetscHashIIterNext(ht, hi);

        /* Loop over all the cells that this entity connects to */
        ierr = DMPlexGetTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
        for ( PetscInt si = 0; si < starSize; si++ ) {
            PetscInt ownedentity = star[2*si];
            /* now loop over all entities in the closure of that cell */
            ierr = DMPlexGetTransitiveClosure(dm, ownedentity, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
            for ( PetscInt ci = 0; ci < closureSize; ci++ ) {
                PetscInt seenentity = closure[2*ci];
                PetscHashIAdd(cht, seenentity, 0);
            }
            ierr = DMPlexRestoreTransitiveClosure(dm, ownedentity, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchGetPointDofs"
/* Given a hash table with a set of topological entities (pts), compute the degrees of
   freedom in global concatenated numbering on those entities.
   For Vanka smoothing, this needs to do something special: ignore dofs of the
   constraint subspace on entities that aren't the base entity we're building the patch
   around. */
static PetscErrorCode PCPatchGetPointDofs(PC_PATCH *patch, PetscHashI pts, PetscHashI dofs, PetscInt base, PetscInt vankaspace)
{
    PetscErrorCode    ierr;
    PetscInt          ldof, loff;
    PetscHashIIter    hi;
    PetscInt          p;

    PetscFunctionBegin;
    PetscHashIClear(dofs);

    for ( PetscInt k = 0; k < patch->nsubspaces; k++ ) {
        PetscSection dofSection = patch->dofSection[k];
        PetscInt bs = patch->bs[k];
        PetscInt subspaceOffset = patch->subspaceOffsets[k];

        if (k == vankaspace) {
            /* only get this subspace dofs at the base entity, not any others */
            ierr = PetscSectionGetDof(dofSection, base, &ldof); CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(dofSection, base, &loff); CHKERRQ(ierr);
            if (0 == ldof) continue;
            for ( PetscInt j = loff; j < ldof + loff; j++ ) {
                for ( PetscInt l = 0; l < bs; l++ ) {
                    PetscInt dof = bs*j + l + subspaceOffset;
                    PetscHashIAdd(dofs, dof, 0);
                }
            }
            continue; /* skip the other dofs of this subspace */
        }

        PetscHashIIterBegin(pts, hi);
        while (!PetscHashIIterAtEnd(pts, hi)) {
            PetscHashIIterGetKey(pts, hi, p);
            PetscHashIIterNext(pts, hi);
            ierr = PetscSectionGetDof(dofSection, p, &ldof); CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(dofSection, p, &loff); CHKERRQ(ierr);
            if (0 == ldof) continue;
            for ( PetscInt j = loff; j < ldof + loff; j++ ) {
                for ( PetscInt l = 0; l < bs; l++ ) {
                    PetscInt dof = bs*j + l + subspaceOffset;
                    PetscHashIAdd(dofs, dof, 0);
                }
            }
        }
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchComputeSetDifference"
/* Given two hash tables A and B, compute the keys in B that are not in A, and
   put them in C */
static PetscErrorCode PCPatchComputeSetDifference(PetscHashI A, PetscHashI B, PetscHashI C)
{
    PetscErrorCode    ierr;
    PetscHashIIter    hi;
    PetscInt          key, val;
    PetscBool         flg;

    PetscFunctionBegin;
    PetscHashIClear(C);

    PetscHashIIterBegin(B, hi);
    while (!PetscHashIIterAtEnd(B, hi)) {
        PetscHashIIterGetKeyVal(B, hi, key, val);
        PetscHashIIterNext(B, hi);
        PetscHashIHasKey(A, key, flg);
        if (!flg) {
            PetscHashIAdd(C, key, val);
        }
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchCompleteCellPatch"
/* On entry, ht contains the topological entities whose dofs we are responsible for solving for;
   on exit, cht contains all the topological entities we need to compute their residuals.
   In full generality this should incorporate knowledge of the sparsity pattern of the matrix;
   here we assume a standard FE sparsity pattern.*/


#undef __FUNCT__
#define __FUNCT__ "PCPatchCreateCellPatches"
/*
 * PCPatchCreateCellPatches - create patches of cells around vertices in the mesh.
 *
 * Input Parameters:
 * + dm - The DMPlex object defining the mesh
 *
 * Output Parameters:
 * + cellCounts - Section with counts of cells around each vertex
 * - cells - IS of the cell point indices of cells in each patch
 */
static PetscErrorCode PCPatchCreateCellPatches(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch      = (PC_PATCH *)pc->data;
    DM              dm;
    DMLabel         ghost;
    PetscInt        pStart, pEnd, vStart, vEnd, cStart, cEnd;
    PetscBool       flg;
    PetscInt        closureSize;
    PetscInt       *closure    = NULL;
    PetscInt       *cellsArray = NULL;
    PetscInt        numCells;
    PetscSection    cellCounts;
    PetscHashI      ht;
    PetscHashI      cht;

    PetscFunctionBegin;

    /* Used to keep track of the cells in the patch. */
    PetscHashICreate(ht);
    PetscHashICreate(cht);

    dm = patch->dm;
    if (!dm) {
        SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC\n");
    }
    ierr = DMPlexGetChart(dm, &pStart, &pEnd); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

    if (patch->codim < 0) { /* codim unset */
        if (patch->dim < 0) { /* dim unset */
            ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
        } else { /* dim set */
            ierr = DMPlexGetDepthStratum(dm, patch->dim, &vStart, &vEnd); CHKERRQ(ierr);
        }
    } else { /* codim set */
        ierr = DMPlexGetHeightStratum(dm, patch->codim, &vStart, &vEnd); CHKERRQ(ierr);
    }

    /* These labels mark the owned points.  We only create patches
     * around entites that this process owns. */
    ierr = DMGetLabel(dm, "pyop2_ghost", &ghost); CHKERRQ(ierr);

    ierr = DMLabelCreateIndex(ghost, pStart, pEnd); CHKERRQ(ierr);

    ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->cellCounts); CHKERRQ(ierr);
    cellCounts = patch->cellCounts;
    ierr = PetscSectionSetChart(cellCounts, vStart, vEnd); CHKERRQ(ierr);

    /* Count cells in the patch surrounding each entity */
    for ( PetscInt v = vStart; v < vEnd; v++ ) {
        PetscHashIIter hi;

        ierr = DMLabelHasPoint(ghost, v, &flg); CHKERRQ(ierr);
        /* Not an owned entity, don't make a cell patch. */
        if (flg) {
            continue;
        }

        ierr = patch->patchconstructop((void*)patch, dm, v, ht); CHKERRQ(ierr);
        ierr = PCPatchCompleteCellPatch(dm, ht, cht);
        PetscHashIIterBegin(cht, hi);
        while (!PetscHashIIterAtEnd(cht, hi)) {
            PetscInt entity;
            PetscHashIIterGetKey(cht, hi, entity);
            if (cStart <= entity && entity < cEnd) {
                ierr = PetscSectionAddDof(cellCounts, v, 1); CHKERRQ(ierr);
            }
            PetscHashIIterNext(cht, hi);
        }
    }
    ierr = DMLabelDestroyIndex(ghost); CHKERRQ(ierr);

    ierr = PetscSectionSetUp(cellCounts); CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(cellCounts, &numCells); CHKERRQ(ierr);
    ierr = PetscMalloc1(numCells, &cellsArray); CHKERRQ(ierr);

    /* Now that we know how much space we need, run through again and
     * actually remember the cells. */
    for ( PetscInt v = vStart; v < vEnd; v++ ) {
        PetscInt ndof, off;
        PetscHashIIter hi;

        ierr = PetscSectionGetDof(cellCounts, v, &ndof); CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cellCounts, v, &off); CHKERRQ(ierr);
        if ( ndof <= 0 ) {
            continue;
        }
        ierr = patch->patchconstructop((void*)patch, dm, v, ht); CHKERRQ(ierr);
        ierr = PCPatchCompleteCellPatch(dm, ht, cht);
        ndof = 0;
        PetscHashIIterBegin(cht, hi);
        while (!PetscHashIIterAtEnd(cht, hi)) {
            PetscInt entity;
            PetscHashIIterGetKey(cht, hi, entity);
            if (cStart <= entity && entity < cEnd) {
                cellsArray[ndof + off] = entity;
                ndof++;
            }
            PetscHashIIterNext(cht, hi);
        }
    }

    ierr = ISCreateGeneral(PETSC_COMM_SELF, numCells, cellsArray, PETSC_OWN_POINTER, &patch->cells); CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd); CHKERRQ(ierr);
    patch->npatch = pEnd - pStart;
    PetscHashIDestroy(ht);
    PetscHashIDestroy(cht);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchCreateCellPatchDiscretisationInfo"
/*
 * PCPatchCreateCellPatchDiscretisationInfo - Build the dof maps for cell patches
 *
 * Input Parameters:
 * + dm - The DMPlex object defining the mesh
 * . cellCounts - Section with counts of cells around each vertex
 * . cells - IS of the cell point indices of cells in each patch
 * . cellNumbering - Section mapping plex cell points to Firedrake cell indices.
 * . nodesPerCell - number of dofs per cell.
 * - cellNodeMap - map from cells to node indices (nodesPerCell * numCells)
 *
 * Output Parameters:
 * + dofs - IS of local dof numbers of each cell in the patch
 * . gtolCounts - Section with counts of dofs per cell patch
 * - gtol - IS mapping from global dofs to local dofs for each patch. 
 */
static PetscErrorCode PCPatchCreateCellPatchDiscretisationInfo(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch           = (PC_PATCH *)pc->data;
    PetscSection    cellCounts      = patch->cellCounts;
    PetscSection    gtolCounts;
    IS              cells           = patch->cells;
    PetscSection    cellNumbering   = patch->cellNumbering;
    PetscInt        numCells;
    PetscInt        numDofs;
    PetscInt        numGlobalDofs;
    PetscInt        totalDofsPerCell = patch->totalDofsPerCell;
    PetscInt        vStart, vEnd;
    const PetscInt *cellsArray;
    PetscInt       *newCellsArray   = NULL;
    PetscInt       *dofsArray       = NULL;
    PetscInt       *asmArray        = NULL;
    PetscInt       *globalDofsArray = NULL;
    PetscInt        globalIndex     = 0;
    PetscHashI      ht;
    PetscFunctionBegin;

    /* dofcounts section is cellcounts section * dofPerCell */
    ierr = PetscSectionGetStorageSize(cellCounts, &numCells); CHKERRQ(ierr);
    numDofs = numCells * totalDofsPerCell;
    ierr = PetscMalloc1(numDofs, &dofsArray); CHKERRQ(ierr);
    ierr = PetscMalloc1(numDofs, &asmArray); CHKERRQ(ierr);
    ierr = PetscMalloc1(numCells, &newCellsArray); CHKERRQ(ierr);
    ierr = PetscSectionGetChart(cellCounts, &vStart, &vEnd); CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCounts); CHKERRQ(ierr);
    gtolCounts = patch->gtolCounts;
    ierr = PetscSectionSetChart(gtolCounts, vStart, vEnd); CHKERRQ(ierr);

    ierr = ISGetIndices(cells, &cellsArray);
    PetscHashICreate(ht);
    for ( PetscInt v = vStart; v < vEnd; v++ ) {
        PetscInt dof, off;
        PetscInt localIndex = 0;
        PetscHashIClear(ht);
        ierr = PetscSectionGetDof(cellCounts, v, &dof); CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cellCounts, v, &off); CHKERRQ(ierr);

        for ( PetscInt k = 0; k < patch->nsubspaces; k++ ) {
            PetscInt nodesPerCell = patch->nodesPerCell[k];
            PetscInt subspaceOffset = patch->subspaceOffsets[k];
            const PetscInt *cellNodeMap = patch->cellNodeMap[k];
            PetscInt bs = patch->bs[k];

            for ( PetscInt i = off; i < off + dof; i++ ) {
                /* Walk over the cells in this patch. */
                const PetscInt c = cellsArray[i];
                PetscInt cell;
                ierr = PetscSectionGetDof(cellNumbering, c, &cell); CHKERRQ(ierr);
                if ( cell <= 0 ) {
                    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                            "Cell doesn't appear in cell numbering map");
                }
                ierr = PetscSectionGetOffset(cellNumbering, c, &cell); CHKERRQ(ierr);
                newCellsArray[i] = cell;
                for ( PetscInt j = 0; j < nodesPerCell; j++ ) {
                    /* For each global dof, map it into contiguous local storage. */
                    const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset;
                    /* finally, loop over block size */
                    for ( PetscInt l = 0; l < bs; l++ ) {
                        PetscInt localDof;
                        PetscHashIMap(ht, globalDof + l, localDof);
                        if (localDof == -1) {
                            localDof = localIndex++;
                            PetscHashIAdd(ht, globalDof + l, localDof);
                        }
                        if ( globalIndex >= numDofs ) {
                            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                                    "Found more dofs than expected");
                        }
                        /* And store. */
                        dofsArray[globalIndex++] = localDof;
                    }
                }
            }
        }
        PetscHashISize(ht, dof);
        /* How many local dofs in this patch? */
        ierr = PetscSectionSetDof(gtolCounts, v, dof); CHKERRQ(ierr);
    }
    if (globalIndex != numDofs) {
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                 "Expected number of dofs (%d) doesn't match found number (%d)",
                 numDofs, globalIndex);
    }
    ierr = PetscSectionSetUp(gtolCounts); CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs); CHKERRQ(ierr);
    ierr = PetscMalloc1(numGlobalDofs, &globalDofsArray); CHKERRQ(ierr);

    /* Now populate the global to local map.  This could be merged
    * into the above loop if we were willing to deal with reallocs. */
    PetscInt key = 0;
    PetscInt asmKey = 0;
    for ( PetscInt v = vStart; v < vEnd; v++ ) {
        PetscInt       dof, off;
        PetscHashIIter hi;
        PetscHashIClear(ht);
        ierr = PetscSectionGetDof(cellCounts, v, &dof); CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cellCounts, v, &off); CHKERRQ(ierr);

        for ( PetscInt k = 0; k < patch->nsubspaces; k++ ) {
            PetscInt nodesPerCell = patch->nodesPerCell[k];
            PetscInt subspaceOffset = patch->subspaceOffsets[k];
            const PetscInt *cellNodeMap = patch->cellNodeMap[k];
            PetscInt bs = patch->bs[k];

            for ( PetscInt i = off; i < off + dof; i++ ) {
                /* Reconstruct mapping of global-to-local on this patch. */
                const PetscInt c = cellsArray[i];
                PetscInt cell;
                ierr = PetscSectionGetOffset(cellNumbering, c, &cell); CHKERRQ(ierr);
                for ( PetscInt j = 0; j < nodesPerCell; j++ ) {
                    for ( PetscInt l = 0; l < bs; l++ ) {
                        const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset + l;
                        const PetscInt localDof = dofsArray[key];
                        key += 1;

                        PetscHashIAdd(ht, globalDof, localDof);
                    }
                }
            }
            if (dof > 0) {
                /* Shove it in the output data structure. */
                PetscInt goff;
                ierr = PetscSectionGetOffset(gtolCounts, v, &goff); CHKERRQ(ierr);
                PetscHashIIterBegin(ht, hi);
                while (!PetscHashIIterAtEnd(ht, hi)) {
                    PetscInt globalDof, localDof;
                    PetscHashIIterGetKeyVal(ht, hi, globalDof, localDof);
                    if (globalDof >= 0) {
                        globalDofsArray[goff + localDof] = globalDof;
                    }
                    PetscHashIIterNext(ht, hi);
                }
            }
        }

        /* At this point, we have a hash table ht built that maps globalDof -> localDof.
           We need to create the dof table laid out cellwise first, then by subspace,
           as the assembler assembles cell-wise and we need to stuff the different
           contributions of the different function spaces to the right places. So we loop
           over cells, then over subspaces. */

        if (patch->nsubspaces > 1) { /* for nsubspaces = 1, data we need is already in dofsArray */
            for (PetscInt i = off; i < off + dof; i++ ) {
                const PetscInt c = cellsArray[i];
                PetscInt cell;
                ierr = PetscSectionGetOffset(cellNumbering, c, &cell); CHKERRQ(ierr);

                for ( PetscInt k = 0; k < patch->nsubspaces; k++ ) {
                    PetscInt nodesPerCell = patch->nodesPerCell[k];
                    PetscInt subspaceOffset = patch->subspaceOffsets[k];
                    const PetscInt *cellNodeMap = patch->cellNodeMap[k];
                    PetscInt bs = patch->bs[k];
                    for ( PetscInt j = 0; j < nodesPerCell; j++ ) {
                        for ( PetscInt l = 0; l < bs; l++ ) {
                            const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset + l;
                            PetscInt localDof;
                            PetscHashIMap(ht, globalDof, localDof);
                            asmArray[asmKey++] = localDof;
                        }
                    }
                }
            }
        }
    }

    if (1 == patch->nsubspaces) { /* replace with memcpy? */
        for (PetscInt i = 0; i < numDofs; i++) {
            asmArray[i] = dofsArray[i];
        }
    }


    PetscHashIDestroy(ht);
    ierr = ISRestoreIndices(cells, &cellsArray);
    ierr = PetscFree(dofsArray); CHKERRQ(ierr);

    /* Replace cell indices with firedrake-numbered ones. */
    ierr = ISGeneralSetIndices(cells, numCells, (const PetscInt *)newCellsArray, PETSC_OWN_POINTER); CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol); CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchCreateCellPatchBCs"
static PetscErrorCode PCPatchCreateCellPatchBCs(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch      = (PC_PATCH *)pc->data;
    DM              dm         = patch->dm;
    PetscInt        numBcs;
    const PetscInt *bcNodes    = NULL;
    PetscSection    gtolCounts = patch->gtolCounts;
    PetscSection    bcCounts;
    PetscSection    multBcCounts;
    IS              gtol = patch->gtol;
    PetscHashI      globalBcs;
    PetscHashI      localBcs;
    PetscHashI      multLocalBcs;
    PetscHashI      patchDofs;
    PetscHashI      ownedpts, seenpts, owneddofs, seendofs, artificialbcs;
    PetscHashIIter  hi;
    PetscInt       *bcsArray   = NULL;
    PetscInt       *multBcsArray   = NULL;
    PetscInt        vStart, vEnd;
    PetscInt        closureSize;
    PetscInt       *closure    = NULL;
    const PetscInt *gtolArray;
    PetscFunctionBegin;

    PetscHashICreate(globalBcs);
    ierr = ISGetIndices(patch->bcNodes, &bcNodes); CHKERRQ(ierr);
    ierr = ISGetSize(patch->bcNodes, &numBcs); CHKERRQ(ierr);
    for ( PetscInt i = 0; i < numBcs; i++ ) {
        PetscHashIAdd(globalBcs, bcNodes[i], 0); /* these are already in concatenated numbering */
    }
    ierr = ISRestoreIndices(patch->bcNodes, &bcNodes); CHKERRQ(ierr);
    PetscHashICreate(patchDofs);
    PetscHashICreate(localBcs);
    PetscHashICreate(multLocalBcs);
    PetscHashICreate(ownedpts);
    PetscHashICreate(seenpts);
    PetscHashICreate(owneddofs);
    PetscHashICreate(seendofs);
    PetscHashICreate(artificialbcs);

    ierr = PetscSectionGetChart(patch->cellCounts, &vStart, &vEnd); CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->bcCounts); CHKERRQ(ierr);
    bcCounts = patch->bcCounts;
    ierr = PetscSectionSetChart(bcCounts, vStart, vEnd); CHKERRQ(ierr);
    ierr = PetscMalloc1(vEnd - vStart, &patch->bcs); CHKERRQ(ierr);

    if (patch->multiplicative) {
        ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->multBcCounts); CHKERRQ(ierr);
        multBcCounts = patch->multBcCounts;
        ierr = PetscSectionSetChart(multBcCounts, vStart, vEnd); CHKERRQ(ierr);
        ierr = PetscMalloc1(vEnd - vStart, &patch->multBcs); CHKERRQ(ierr);
    }

    ierr = ISGetIndices(gtol, &gtolArray); CHKERRQ(ierr);
    for ( PetscInt v = vStart; v < vEnd; v++ ) {
        PetscInt numBcs, dof, off;
        PetscInt bcIndex = 0;
        PetscInt multBcIndex = 0;
        PetscHashIClear(patchDofs);
        PetscHashIClear(localBcs);
        PetscHashIClear(multLocalBcs);
        ierr = PetscSectionGetDof(gtolCounts, v, &dof); CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(gtolCounts, v, &off); CHKERRQ(ierr);
        for ( PetscInt i = off; i < off + dof; i++ ) {
            PetscBool flg;
            const PetscInt globalDof = gtolArray[i];
            const PetscInt localDof = i - off;
            PetscHashIAdd(patchDofs, globalDof, localDof);
            PetscHashIHasKey(globalBcs, globalDof, flg);
            if (flg) {
                PetscHashIAdd(localBcs, localDof, 0);
                PetscHashIAdd(multLocalBcs, localDof, 0);
            }
        }

        /* If we're doing multiplicative, make the BC data structures now
           corresponding solely to actual globally imposed Dirichlet BCs */
        if (patch->multiplicative) {
            PetscHashISize(multLocalBcs, numBcs);
            ierr = PetscSectionSetDof(multBcCounts, v, numBcs); CHKERRQ(ierr);
            ierr = PetscMalloc1(numBcs, &multBcsArray); CHKERRQ(ierr);
            ierr = PetscHashIGetKeys(multLocalBcs, &multBcIndex, multBcsArray); CHKERRQ(ierr);
            ierr = PetscSortInt(numBcs, multBcsArray); CHKERRQ(ierr);
            ierr = ISCreateGeneral(PETSC_COMM_SELF, numBcs, multBcsArray, PETSC_OWN_POINTER, &(patch->multBcs[v - vStart])); CHKERRQ(ierr);
        }

        /* Now figure out the artificial BCs: the set difference of {dofs on entities
           I see on the patch}\{dofs I am responsible for updating} */
        ierr = patch->patchconstructop((void*)patch, dm, v, ownedpts); CHKERRQ(ierr);
        ierr = PCPatchCompleteCellPatch(dm, ownedpts, seenpts); CHKERRQ(ierr);
        ierr = PCPatchGetPointDofs(patch, ownedpts, owneddofs, v, patch->vankaspace); CHKERRQ(ierr);
        ierr = PCPatchGetPointDofs(patch, seenpts, seendofs, v, -1); CHKERRQ(ierr);
        ierr = PCPatchComputeSetDifference(owneddofs, seendofs, artificialbcs); CHKERRQ(ierr);

        PetscHashIIterBegin(artificialbcs, hi);
        while (!PetscHashIIterAtEnd(artificialbcs, hi)) {
            PetscInt globalDof, localDof;
            PetscHashIIterGetKey(artificialbcs, hi, globalDof);
            PetscHashIIterNext(artificialbcs, hi);
            PetscHashIMap(patchDofs, globalDof, localDof);
            if ( localDof == -1 ) {
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                        "Didn't find dof in patch\n");
            }
            PetscHashIAdd(localBcs, localDof, 0);
        }

        /* OK, now we have a hash table with all the bcs indicated by
         * the artificial and global bcs */
        PetscHashISize(localBcs, numBcs);
        ierr = PetscSectionSetDof(bcCounts, v, numBcs); CHKERRQ(ierr);
        ierr = PetscMalloc1(numBcs, &bcsArray); CHKERRQ(ierr);
        ierr = PetscHashIGetKeys(localBcs, &bcIndex, bcsArray); CHKERRQ(ierr);
        ierr = PetscSortInt(numBcs, bcsArray); CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF, numBcs, bcsArray, PETSC_OWN_POINTER, &(patch->bcs[v - vStart])); CHKERRQ(ierr);
    }
    if (closure) {
        ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(gtol, &gtolArray); CHKERRQ(ierr);
    PetscHashIDestroy(artificialbcs);
    PetscHashIDestroy(seendofs);
    PetscHashIDestroy(owneddofs);
    PetscHashIDestroy(seenpts);
    PetscHashIDestroy(ownedpts);
    PetscHashIDestroy(localBcs);
    PetscHashIDestroy(multLocalBcs);
    PetscHashIDestroy(patchDofs);
    PetscHashIDestroy(globalBcs);

    ierr = PetscSectionSetUp(bcCounts); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_PATCH"
static PetscErrorCode PCReset_PATCH(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscInt        i;

    PetscFunctionBegin;
    ierr = DMDestroy(&patch->dm); CHKERRQ(ierr);
    ierr = PetscSFDestroy(&patch->defaultSF); CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&patch->cellCounts); CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&patch->cellNumbering); CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&patch->gtolCounts); CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&patch->bcCounts); CHKERRQ(ierr);
    ierr = ISDestroy(&patch->gtol); CHKERRQ(ierr);
    ierr = ISDestroy(&patch->cells); CHKERRQ(ierr);
    ierr = ISDestroy(&patch->dofs); CHKERRQ(ierr);
    ierr = ISDestroy(&patch->bcNodes); CHKERRQ(ierr);

    if (patch->dofSection) {
        for (i = 0; i < patch->nsubspaces; i++) {
            ierr = PetscSectionDestroy(&patch->dofSection[i]); CHKERRQ(ierr);
        }
    }
    ierr = PetscFree(patch->dofSection); CHKERRQ(ierr);
    ierr = PetscFree(patch->bs); CHKERRQ(ierr);
    ierr = PetscFree(patch->nodesPerCell); CHKERRQ(ierr);
    ierr = PetscFree(patch->cellNodeMap); CHKERRQ(ierr);
    ierr = PetscFree(patch->bcNodes); CHKERRQ(ierr);

    if (patch->bcs) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = ISDestroy(&patch->bcs[i]); CHKERRQ(ierr);
        }
        ierr = PetscFree(patch->bcs); CHKERRQ(ierr);
    }

    if (patch->multiplicative) {
        ierr = PetscSectionDestroy(&patch->multBcCounts); CHKERRQ(ierr);
        if (patch->multBcs) {
            for ( i = 0; i < patch->npatch; i++ ) {
                ierr = ISDestroy(&patch->multBcs[i]); CHKERRQ(ierr);
            }
            ierr = PetscFree(patch->multBcs); CHKERRQ(ierr);
        }
    }

    if (patch->free_type) {
        ierr = MPI_Type_free(&patch->data_type); CHKERRQ(ierr);
        patch->data_type = MPI_DATATYPE_NULL; 
    }

    if (patch->ksp) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = KSPReset(patch->ksp[i]); CHKERRQ(ierr);
        }
    }

    ierr = VecDestroy(&patch->localX); CHKERRQ(ierr);
    ierr = VecDestroy(&patch->localY); CHKERRQ(ierr);
    if (patch->patchX) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = VecDestroy(patch->patchX + i); CHKERRQ(ierr);
        }
        ierr = PetscFree(patch->patchX); CHKERRQ(ierr);
    }
    if (patch->patchY) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = VecDestroy(patch->patchY + i); CHKERRQ(ierr);
        }
        ierr = PetscFree(patch->patchY); CHKERRQ(ierr);
    }
    if (patch->mat) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = MatDestroy(patch->mat + i); CHKERRQ(ierr);
            if (patch->multiplicative) {
                ierr = MatDestroy(patch->multMat + i); CHKERRQ(ierr);
            }
        }
        ierr = PetscFree(patch->mat); CHKERRQ(ierr);
        if (patch->multiplicative) {
            ierr = PetscFree(patch->multMat); CHKERRQ(ierr);
        }
    }
    ierr = PetscFree(patch->sub_mat_type); CHKERRQ(ierr);

    patch->free_type = PETSC_FALSE;
    patch->bs = 0;
    patch->cellNodeMap = NULL;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_PATCH"
static PetscErrorCode PCDestroy_PATCH(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscInt        i;

    PetscFunctionBegin;

    ierr = PCReset_PATCH(pc); CHKERRQ(ierr);
    if (patch->ksp) {
        for ( i = 0; i < patch->npatch; i++ ) {
            ierr = KSPDestroy(&patch->ksp[i]); CHKERRQ(ierr);
        }
        ierr = PetscFree(patch->ksp); CHKERRQ(ierr);
    }
    ierr = PetscFree(pc->data); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchCreateMatrix"
static PetscErrorCode PCPatchCreateMatrix(PC pc, Vec x, Vec y, Mat *mat)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscInt        csize, rsize, cbs, rbs;
    const char     *prefix = NULL;

    PetscFunctionBegin;
    ierr = VecGetSize(x, &csize); CHKERRQ(ierr);
    ierr = VecGetBlockSize(x, &cbs); CHKERRQ(ierr);
    ierr = VecGetSize(y, &rsize); CHKERRQ(ierr);
    ierr = VecGetBlockSize(y, &rbs); CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF, mat); CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc, &prefix); CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(*mat, prefix); CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(*mat, "sub_"); CHKERRQ(ierr);
    if (patch->sub_mat_type) {
        ierr = MatSetType(*mat, patch->sub_mat_type); CHKERRQ(ierr);
    }
    ierr = MatSetSizes(*mat, rsize, csize, rsize, csize); CHKERRQ(ierr);
    ierr = MatSetUp(*mat); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchComputeOperator"
static PetscErrorCode PCPatchComputeOperator(PC pc, Mat mat, Mat multMat, PetscInt which)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    const PetscInt *dofsArray;
    const PetscInt *cellsArray;
    PetscInt        ncell, offset, pStart, pEnd;
    PetscInt i = which;

    PetscFunctionBegin;

    ierr = PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0); CHKERRQ(ierr);
    if (!patch->usercomputeop) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback\n");
    }
    ierr = ISGetIndices(patch->dofs, &dofsArray); CHKERRQ(ierr);
    ierr = ISGetIndices(patch->cells, &cellsArray); CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd); CHKERRQ(ierr);

    which += pStart;
    if (which >= pEnd) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for operator index is invalid\n"); CHKERRQ(ierr);
    }

    ierr = PetscSectionGetDof(patch->cellCounts, which, &ncell); CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->cellCounts, which, &offset); CHKERRQ(ierr);
    PetscStackPush("PCPatch user callback");
    ierr = patch->usercomputeop(pc, mat, ncell, cellsArray + offset, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, patch->usercomputectx); CHKERRQ(ierr);
    PetscStackPop;
    ierr = ISRestoreIndices(patch->dofs, &dofsArray); CHKERRQ(ierr);
    ierr = ISRestoreIndices(patch->cells, &cellsArray); CHKERRQ(ierr);

    /* Apply boundary conditions.  Could also do this through the local_to_patch guy. */
    if (patch->multiplicative) {
        ierr = MatCopy(mat, multMat, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatZeroRowsColumnsIS(multMat, patch->multBcs[which-pStart], (PetscScalar)1.0, NULL, NULL); CHKERRQ(ierr);
    }
    ierr = MatZeroRowsColumnsIS(mat, patch->bcs[which-pStart], (PetscScalar)1.0, NULL, NULL); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatch_ScatterLocal_Private"
static PetscErrorCode PCPatch_ScatterLocal_Private(PC pc, PetscInt p,
                                                   Vec x, Vec y,
                                                   InsertMode mode,
                                                   ScatterMode scat)
{
    PetscErrorCode ierr;
    PC_PATCH          *patch   = (PC_PATCH *)pc->data;
    const PetscScalar *xArray = NULL;
    PetscScalar *yArray = NULL;
    const PetscInt *gtolArray = NULL;
    PetscInt offset, size;

    PetscFunctionBeginHot;
    ierr = PetscLogEventBegin(PC_Patch_Scatter, pc, 0, 0, 0); CHKERRQ(ierr);

    ierr = VecGetArrayRead(x, &xArray); CHKERRQ(ierr);
    ierr = VecGetArray(y, &yArray); CHKERRQ(ierr);

    ierr = PetscSectionGetDof(patch->gtolCounts, p, &size); CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->gtolCounts, p, &offset); CHKERRQ(ierr);
    ierr = ISGetIndices(patch->gtol, &gtolArray); CHKERRQ(ierr);
    if (mode == INSERT_VALUES && scat != SCATTER_FORWARD) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't insert if not scattering forward\n");
    }
    if (mode == ADD_VALUES && scat != SCATTER_REVERSE) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't add if not scattering reverse\n");
    }
    for ( PetscInt lidx = 0; lidx < size; lidx++ ) {
        const PetscInt gidx = gtolArray[lidx + offset];
        if (mode == INSERT_VALUES) {
            yArray[lidx] = xArray[gidx];
        } else {
            yArray[gidx] += xArray[lidx];
        }
    }
    ierr = VecRestoreArrayRead(x, &xArray); CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &yArray); CHKERRQ(ierr);
    ierr = ISRestoreIndices(patch->gtol, &gtolArray); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_Patch_Scatter, pc, 0, 0, 0); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_PATCH"
static PetscErrorCode PCSetUp_PATCH(PC pc)
{
    PetscErrorCode  ierr;
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    const char     *prefix;
    PetscScalar    *patchX  = NULL;
    PetscInt       pStart, numBcs;
    const PetscInt *bcNodes = NULL;

    PetscFunctionBegin;

    if (!pc->setupcalled) {
        PetscInt     pStart, pEnd;
        PetscInt     localSize;
        ierr = PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0); CHKERRQ(ierr);

        /* PEF: disabling parallel for now */
        /* switch (patch->bs) {
        case 1:
            patch->data_type = MPIU_SCALAR;
            break;
        case 2:
            patch->data_type = MPIU_2SCALAR;
            break;
        default:
            ierr = MPI_Type_contiguous(patch->bs, MPIU_SCALAR, &patch->data_type); CHKERRQ(ierr);
            ierr = MPI_Type_commit(&patch->data_type); CHKERRQ(ierr);
            patch->free_type = PETSC_TRUE;
        } */
        patch->free_type = PETSC_FALSE;

        localSize = patch->subspaceOffsets[patch->nsubspaces];
        ierr = VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localX); CHKERRQ(ierr);
        ierr = VecSetUp(patch->localX); CHKERRQ(ierr);
        ierr = VecDuplicate(patch->localX, &patch->localY); CHKERRQ(ierr);
        ierr = PCPatchCreateCellPatches(pc); CHKERRQ(ierr);
        ierr = PCPatchCreateCellPatchDiscretisationInfo(pc); CHKERRQ(ierr);
        ierr = PCPatchCreateCellPatchBCs(pc); CHKERRQ(ierr);

        /* OK, now build the work vectors */
        ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd); CHKERRQ(ierr);
        ierr = PetscMalloc1(patch->npatch, &patch->patchX); CHKERRQ(ierr);
        ierr = PetscMalloc1(patch->npatch, &patch->patchY); CHKERRQ(ierr);
        for ( PetscInt i = pStart; i < pEnd; i++ ) {
            PetscInt dof;
            ierr = PetscSectionGetDof(patch->gtolCounts, i, &dof); CHKERRQ(ierr);
            ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchX[i - pStart]); CHKERRQ(ierr);
            ierr = VecSetUp(patch->patchX[i - pStart]); CHKERRQ(ierr);
            ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchY[i - pStart]); CHKERRQ(ierr);
            ierr = VecSetUp(patch->patchY[i - pStart]); CHKERRQ(ierr);
        }
        ierr = PetscMalloc1(patch->npatch, &patch->ksp); CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc, &prefix); CHKERRQ(ierr);
        for ( PetscInt i = 0; i < patch->npatch; i++ ) {
            ierr = KSPCreate(PETSC_COMM_SELF, patch->ksp + i); CHKERRQ(ierr);
            ierr = KSPSetOptionsPrefix(patch->ksp[i], prefix); CHKERRQ(ierr);
            ierr = KSPAppendOptionsPrefix(patch->ksp[i], "sub_"); CHKERRQ(ierr);
        }
        if (patch->save_operators) {
            ierr = PetscMalloc1(patch->npatch, &patch->mat); CHKERRQ(ierr);
            if (patch->multiplicative) {
                ierr = PetscMalloc1(patch->npatch, &patch->multMat); CHKERRQ(ierr);
            }
            for ( PetscInt i = 0; i < patch->npatch; i++ ) {
                ierr = PCPatchCreateMatrix(pc, patch->patchX[i], patch->patchY[i], patch->mat + i); CHKERRQ(ierr);
                if (patch->multiplicative) {
                    ierr = PCPatchCreateMatrix(pc, patch->patchX[i], patch->patchY[i], patch->multMat + i); CHKERRQ(ierr);
                }
            }
        }
        ierr = PetscLogEventEnd(PC_Patch_CreatePatches, pc, 0, 0, 0); CHKERRQ(ierr);
    }

    /* If desired, calculate weights for dof multiplicity */

    if (patch->partition_of_unity) {
        ierr = VecDuplicate(patch->localX, &patch->dof_weights); CHKERRQ(ierr);
        ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL); CHKERRQ(ierr);
        for ( PetscInt i = 0; i < patch->npatch; i++ ) {
            ierr = VecSet(patch->patchX[i], 1.0); CHKERRQ(ierr);
            /* TODO: Do we need different scatters for X and Y? */
            ierr = VecGetArray(patch->patchX[i], &patchX); CHKERRQ(ierr);
            /* Apply bcs to patchX (zero entries) */
            ierr = ISGetLocalSize(patch->bcs[i], &numBcs); CHKERRQ(ierr);
            ierr = ISGetIndices(patch->bcs[i], &bcNodes); CHKERRQ(ierr);
            for ( PetscInt j = 0; j < numBcs; j++ ) {
                const PetscInt idx = bcNodes[j];
                patchX[idx] = 0;
            }
            ierr = ISRestoreIndices(patch->bcs[i], &bcNodes); CHKERRQ(ierr);
            ierr = VecRestoreArray(patch->patchX[i], &patchX); CHKERRQ(ierr);

            ierr = PCPatch_ScatterLocal_Private(pc, i + pStart,
                                                patch->patchX[i], patch->dof_weights,
                                                ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
        }
        ierr = VecReciprocal(patch->dof_weights); CHKERRQ(ierr);
    }

    if (patch->save_operators) {
        for ( PetscInt i = 0; i < patch->npatch; i++ ) {
            ierr = MatZeroEntries(patch->mat[i]); CHKERRQ(ierr);
            if (patch->multiplicative) {
                ierr = PCPatchComputeOperator(pc, patch->mat[i], patch->multMat[i], i); CHKERRQ(ierr);
            } else {
                ierr = PCPatchComputeOperator(pc, patch->mat[i], NULL, i); CHKERRQ(ierr);
            }
            ierr = KSPSetOperators(patch->ksp[i], patch->mat[i], patch->mat[i]); CHKERRQ(ierr);
        }
    }
    if (!pc->setupcalled) {
        for ( PetscInt i = 0; i < patch->npatch; i++ ) {
            ierr = KSPSetFromOptions(patch->ksp[i]); CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_PATCH"
static PetscErrorCode PCApply_PATCH(PC pc, Vec x, Vec y)
{
    PetscErrorCode     ierr;
    PC_PATCH          *patch   = (PC_PATCH *)pc->data;
    const PetscScalar *globalX = NULL;
    PetscScalar       *localX  = NULL;
    PetscScalar       *localY  = NULL;
    PetscScalar       *globalY = NULL;
    PetscScalar       *patchX  = NULL;
    const PetscInt    *bcNodes = NULL;
    PetscInt           pStart, numBcs, size;
    PetscFunctionBegin;

    ierr = PetscLogEventBegin(PC_Patch_Apply, pc, 0, 0, 0); CHKERRQ(ierr);
    ierr = PetscOptionsPushGetViewerOff(PETSC_TRUE); CHKERRQ(ierr);
    /* Scatter from global space into overlapped local spaces */
    /* ierr = VecGetArrayRead(x, &globalX); CHKERRQ(ierr);
    ierr = VecGetArray(patch->localX, &localX); CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(patch->defaultSF, patch->data_type, globalX, localX); CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(patch->defaultSF, patch->data_type, globalX, localX); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x, &globalX); CHKERRQ(ierr);
    ierr = VecRestoreArray(patch->localX, &localX); CHKERRQ(ierr); */
    /* PEF: for now, don't care about parallel: veccopy globalX -> localX */
    ierr = VecCopy(x, patch->localX); CHKERRQ(ierr);

    ierr = VecSet(patch->localY, 0.0); CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL); CHKERRQ(ierr);
    for ( PetscInt i = 0; i < patch->npatch; i++ ) {
        PetscInt start, len;
        Mat multMat = NULL;

        ierr = PetscSectionGetDof(patch->gtolCounts, i + pStart, &len); CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->gtolCounts, i + pStart, &start); CHKERRQ(ierr);
        if ( len <= 0 ) {
            /* TODO: Squash out these guys in the setup as well. */
            continue;
        }
        ierr = PCPatch_ScatterLocal_Private(pc, i + pStart,
                                            patch->localX, patch->patchX[i],
                                            INSERT_VALUES,
                                            SCATTER_FORWARD); CHKERRQ(ierr);
        /* TODO: Do we need different scatters for X and Y? */
        ierr = VecGetArray(patch->patchX[i], &patchX); CHKERRQ(ierr);
        /* Apply bcs to patchX (zero entries) */
        ierr = ISGetLocalSize(patch->bcs[i], &numBcs); CHKERRQ(ierr);
        ierr = ISGetIndices(patch->bcs[i], &bcNodes); CHKERRQ(ierr);
        for ( PetscInt j = 0; j < numBcs; j++ ) {
            const PetscInt idx = bcNodes[j];
            patchX[idx] = 0;
        }
        ierr = ISRestoreIndices(patch->bcs[i], &bcNodes); CHKERRQ(ierr);
        ierr = VecRestoreArray(patch->patchX[i], &patchX); CHKERRQ(ierr);
        if (!patch->save_operators) {
            Mat mat;
            ierr = PCPatchCreateMatrix(pc, patch->patchX[i], patch->patchY[i], &mat); CHKERRQ(ierr);
            if (patch->multiplicative) {
                ierr = PCPatchCreateMatrix(pc, patch->patchX[i], patch->patchY[i], &multMat); CHKERRQ(ierr);
            }
            /* Populate operator here. */
            ierr = PCPatchComputeOperator(pc, mat, multMat, i); CHKERRQ(ierr);
            ierr = KSPSetOperators(patch->ksp[i], mat, mat);
            /* Drop reference so the KSPSetOperators below will blow it away. */
            ierr = MatDestroy(&mat); CHKERRQ(ierr);
        }

        if (patch->save_operators && patch->multiplicative) {
            multMat = patch->multMat[i];
        }

        ierr = PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0); CHKERRQ(ierr);
        ierr = KSPSolve(patch->ksp[i], patch->patchX[i], patch->patchY[i]); CHKERRQ(ierr);
        ierr = PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0); CHKERRQ(ierr);
        if (!patch->save_operators) {
            PC pc;
            ierr = KSPSetOperators(patch->ksp[i], NULL, NULL); CHKERRQ(ierr);
            ierr = KSPGetPC(patch->ksp[i], &pc); CHKERRQ(ierr);
            /* Destroy PC context too, otherwise the factored matrix hangs around. */
            ierr = PCReset(pc); CHKERRQ(ierr);
        }

        ierr = PCPatch_ScatterLocal_Private(pc, i + pStart,
                                            patch->patchY[i], patch->localY,
                                            ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

        /* Get the matrix on the patch but with only global bcs applied.
         * This matrix is then multiplied with the result from the previous solve
         * to obtain by how much the residual changes. */

        if(patch->multiplicative){
            ierr = MatMult(multMat, patch->patchY[i], patch->patchX[i]); CHKERRQ(ierr);
            ierr = VecScale(patch->patchX[i], -1.0); CHKERRQ(ierr);
            ierr = PCPatch_ScatterLocal_Private(pc, i + pStart,
                                                patch->patchX[i], patch->localX,
                                                ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
        }

        if (patch->multiplicative && !patch->save_operators) {
            ierr = MatDestroy(&multMat); CHKERRQ(ierr);
        }
    }
    /* Now patch->localY contains the solution of the patch solves, so
     * we need to combine them all.  This hardcodes an ADDITIVE
     * combination right now.  If one wanted multiplicative, the
     * scatter/gather stuff would have to be reworked a bit. */
    ierr = VecSet(y, 0.0); CHKERRQ(ierr);
    /* PEF: replace with VecCopy for now */
    ierr = VecCopy(patch->localY, y); CHKERRQ(ierr);
    ierr = VecGetArray(y, &globalY); CHKERRQ(ierr);
    /* ierr = VecGetArrayRead(patch->localY, (const PetscScalar **)&localY); CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(patch->defaultSF, patch->data_type, localY, globalY, MPI_SUM); CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(patch->defaultSF, patch->data_type, localY, globalY, MPI_SUM); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(patch->localY, (const PetscScalar **)&localY); CHKERRQ(ierr); */

    if (patch->partition_of_unity) {
        ierr = VecRestoreArray(y, &globalY); CHKERRQ(ierr);
        ierr = VecPointwiseMult(y, y, patch->dof_weights); CHKERRQ(ierr);
        ierr = VecGetArray(y, &globalY); CHKERRQ(ierr);
    }

    /* Now we need to send the global BC values through */
    ierr = VecGetArrayRead(x, &globalX); CHKERRQ(ierr);
    ierr = ISGetSize(patch->bcNodes, &numBcs); CHKERRQ(ierr);
    ierr = ISGetIndices(patch->bcNodes, &bcNodes); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &size); CHKERRQ(ierr);
    for ( PetscInt i = 0; i < numBcs; i++ ) {
        const PetscInt idx = bcNodes[i];
        if (idx < size) {
            globalY[idx] = globalX[idx];
        }
    }

    ierr = ISRestoreIndices(patch->bcNodes, &bcNodes); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x, &globalX); CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &globalY); CHKERRQ(ierr);
    ierr = PetscOptionsPopGetViewerOff(); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_Patch_Apply, pc, 0, 0, 0); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUpOnBlocks_PATCH"
static PetscErrorCode PCSetUpOnBlocks_PATCH(PC pc)
{
  PC_PATCH           *patch = (PC_PATCH*)pc->data;
  PetscErrorCode      ierr;
  PetscInt            i;
  KSPConvergedReason  reason;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
  for (i=0; i<patch->npatch; i++) {
    ierr = KSPSetUp(patch->ksp[i]); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(patch->ksp[i], &reason); CHKERRQ(ierr);
    if (reason == KSP_DIVERGED_PCSETUP_FAILED) {
      pc->failedreason = PC_SUBPC_ERROR;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchConstruct_Current"
PETSC_EXTERN PetscErrorCode PCPatchConstruct_Current(void *vpatch, DM dm, PetscInt entity, PetscHashI ht)
{
    PetscErrorCode ierr;
    PetscInt       starSize, closureSize, fclosureSize;
    PetscInt      *star = NULL, *closure = NULL, *fclosure = NULL;
    PetscInt       vStart, vEnd;
    PetscInt       fStart, fEnd;

    PetscFunctionBegin;
    PetscHashIClear(ht);

    /* Find out what the facets and vertices are */
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm,  0, &vStart, &vEnd); CHKERRQ(ierr);

    /* To start with, add the entity we care about */
    PetscHashIAdd(ht, entity, 0);

    /* Loop over all the cells that this entity connects to */
    ierr = DMPlexGetTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
    for ( PetscInt si = 0; si < starSize; si++ ) {
        PetscInt cell = star[2*si];
        /* now loop over all entities in the closure of that cell */
        ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
        for ( PetscInt ci = 0; ci < closureSize; ci++ ) {
            PetscInt newentity = closure[2*ci];

            if (vStart <= newentity && newentity < vEnd) {
                /* Is another vertex. Don't want to add. continue */
                continue;
            } else if (fStart <= newentity && newentity < fEnd) {
                /* Is another facet. Need to check if our own vertex is in its closure */
                PetscBool should_add = PETSC_FALSE;
                ierr = DMPlexGetTransitiveClosure(dm, newentity, PETSC_TRUE, &fclosureSize, &fclosure); CHKERRQ(ierr);
                for ( PetscInt fi = 0; fi < fclosureSize; fi++) {
                    PetscInt fentity = fclosure[2*fi];
                    if (entity == fentity) {
                        should_add = PETSC_TRUE;
                        break;
                    }
                }
                ierr = DMPlexRestoreTransitiveClosure(dm, newentity, PETSC_TRUE, &fclosureSize, &fclosure); CHKERRQ(ierr);
                if (should_add) {
                    PetscHashIAdd(ht, newentity, 0);
                }
            } else { /* not a vertex, not a facet, just add */
                PetscHashIAdd(ht, newentity, 0);
            }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPatchConstruct_Vanka"
PETSC_EXTERN PetscErrorCode PCPatchConstruct_Vanka(void *vpatch, DM dm, PetscInt entity, PetscHashI ht)
{
    PetscErrorCode ierr;
    PC_PATCH      *patch = (PC_PATCH*) vpatch;
    PetscInt       starSize, closureSize;
    PetscInt      *star = NULL, *closure = NULL;
    PetscInt       iStart, iEnd;
    PetscBool      shouldIgnore;

    PetscFunctionBegin;
    PetscHashIClear(ht);

    /* To start with, add the entity we care about */
    PetscHashIAdd(ht, entity, 0);

    /* Should we ignore any topological entities of a certain dimension? */
    if (patch->vankadim >= 0) {
        shouldIgnore = PETSC_TRUE;
        ierr = DMPlexGetDepthStratum(dm, patch->vankadim, &iStart, &iEnd); CHKERRQ(ierr);
    } else {
        shouldIgnore = PETSC_FALSE;
    }

    /* Loop over all the cells that this entity connects to */
    ierr = DMPlexGetTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
    for ( PetscInt si = 0; si < starSize; si++ ) {
        PetscInt cell = star[2*si];
        /* now loop over all entities in the closure of that cell */
        ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
        for ( PetscInt ci = 0; ci < closureSize; ci++ ) {
            PetscInt newentity = closure[2*ci];
            if (shouldIgnore && iStart <= newentity && newentity < iEnd) {
                /* We've been told to ignore entities of this type.*/
                continue;
            }
            PetscHashIAdd(ht, newentity, 0);
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure); CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, entity, PETSC_FALSE, &starSize, &star); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_PATCH"
static PetscErrorCode PCSetFromOptions_PATCH(PetscOptionItems *PetscOptionsObject, PC pc)
{
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscErrorCode  ierr;
    PetscBool       flg, dimflg, codimflg;
    char            sub_mat_type[256];
    const char     *patchConstructionTypes[2]  = {"current", "vanka"};
    PetscInt        patchConstructionType;

    PetscFunctionBegin;
    ierr = PetscOptionsHead(PetscOptionsObject, "Vertex-patch Additive Schwarz options"); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-pc_patch_save_operators", "Store all patch operators for lifetime of PC?",
                            "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-pc_patch_partition_of_unity", "Weight contributions by dof multiplicity?",
                            "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg); CHKERRQ(ierr);

    ierr = PetscOptionsBool("-pc_patch_multiplicative", "Gauss-Seidel instead of Jacobi?",
                            "PCPatchSetMultiplicative", patch->multiplicative, &patch->multiplicative, &flg); CHKERRQ(ierr);

    ierr = PetscOptionsInt("-pc_patch_construction_dim", "What dimension of entity to construct patches by? (0 = vertices)", "PCSetFromOptions_PATCH", patch->dim, &patch->dim, &dimflg);
    ierr = PetscOptionsInt("-pc_patch_construction_codim", "What co-dimension of entity to construct patches by? (0 = cells)", "PCSetFromOptions_PATCH", patch->codim, &patch->codim, &codimflg);
    if (dimflg && codimflg) {
        SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Can only set one of dimension or co-dimension");
    }
    ierr = PetscOptionsEList("-pc_patch_construction_type", "How should the patches be constructed?", "PCSetFromOptions_PATCH", patchConstructionTypes, 2, patchConstructionTypes[0], &patchConstructionType, &flg);
    if (flg) {
        if (patchConstructionType == 0) {
            patch->patchconstructop = PCPatchConstruct_Current;
        } else if (patchConstructionType == 1) {
            patch->patchconstructop = PCPatchConstruct_Vanka;
            ierr = PetscOptionsInt("-pc_patch_vanka_dim", "Topological dimension of entities for Vanka to ignore", "PCSetFromOptions_PATCH", patch->vankadim, &patch->vankadim, &flg);
            ierr = PetscOptionsInt("-pc_patch_vanka_space", "What subspace is the constraint space for Vanka?", "PCSetFromOptions_PATCH", patch->vankaspace, &patch->vankaspace, &flg);
            if (!flg) {
                SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Must set Vanka subspace using -pc_patch_vanka_space when using Vanka");
            }
        } else {
            SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Unknown patch construction type");
        }
    }

    ierr = PetscOptionsFList("-pc_patch_sub_mat_type", "Matrix type for patch solves", "PCPatchSetSubMatType",MatList, NULL, sub_mat_type, 256, &flg); CHKERRQ(ierr);
    if (flg) {
        ierr = PCPatchSetSubMatType(pc, sub_mat_type); CHKERRQ(ierr);
    }
    ierr = PetscOptionsTail(); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_PATCH"
static PetscErrorCode PCView_PATCH(PC pc, PetscViewer viewer)
{
    PC_PATCH       *patch = (PC_PATCH *)pc->data;
    PetscErrorCode  ierr;
    PetscMPIInt     rank;
    PetscBool       isascii;
    PetscViewer     sviewer;
    PetscFunctionBegin;
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
    if (!isascii) {
        PetscFunctionReturn(0);
    }
    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Vertex-patch Additive Schwarz with %d patches\n", patch->npatch); CHKERRQ(ierr);
    if (!patch->save_operators) {
        ierr = PetscViewerASCIIPrintf(viewer, "Not saving patch operators (rebuilt every PCApply)\n"); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer, "Saving patch operators (rebuilt every PCSetUp)\n"); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "DM used to define patches:\n"); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
    if (patch->dm) {
        ierr = DMView(patch->dm, viewer); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer, "DM not yet set.\n"); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "KSP on patches (all same):\n"); CHKERRQ(ierr);

    if (patch->ksp) {
        ierr = PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer); CHKERRQ(ierr);
        if (!rank) {
            ierr = PetscViewerASCIIPushTab(sviewer); CHKERRQ(ierr);
            ierr = KSPView(patch->ksp[0], sviewer); CHKERRQ(ierr);
            ierr = PetscViewerASCIIPopTab(sviewer); CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer); CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "KSP not yet set.\n"); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    }

    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreate_PATCH"
PETSC_EXTERN PetscErrorCode PCCreate_PATCH(PC pc)
{
    PetscErrorCode ierr;
    PC_PATCH       *patch;

    PetscFunctionBegin;

    ierr = PetscNewLog(pc, &patch); CHKERRQ(ierr);

    /* Set some defaults */
    patch->sub_mat_type       = NULL;
    patch->save_operators     = PETSC_TRUE;
    patch->partition_of_unity = PETSC_FALSE;
    patch->multiplicative     = PETSC_FALSE;
    patch->codim              = -1;
    patch->dim                = -1;
    patch->vankaspace         = -1;
    patch->vankadim           = -1;
    patch->patchconstructop   = PCPatchConstruct_Current;

    pc->data                 = (void *)patch;
    pc->ops->apply           = PCApply_PATCH;
    pc->ops->applytranspose  = 0; /* PCApplyTranspose_PATCH; */
    pc->ops->setup           = PCSetUp_PATCH;
    pc->ops->reset           = PCReset_PATCH;
    pc->ops->destroy         = PCDestroy_PATCH;
    pc->ops->setfromoptions  = PCSetFromOptions_PATCH;
    pc->ops->setuponblocks   = PCSetUpOnBlocks_PATCH;
    pc->ops->view            = PCView_PATCH;
    pc->ops->applyrichardson = 0;

    PetscFunctionReturn(0);
}
