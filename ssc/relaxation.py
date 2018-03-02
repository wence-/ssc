import numpy
from itertools import chain
from firedrake.petsc import PETSc

__all__ = ['PlaneSmoother', 'OrderedVanka', 'OrderedStar']

class PlaneSmoother(object):
    @staticmethod
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    def sort_entities(self, dm, axis, dir, ndiv):
        # compute
        # [(pStart, (x, y, z)), (pEnd, (x, y, z))]
        entities = [(p, self.coords(dm, p)) for p in range(*dm.getChart())]

        minx = min(entities, key=lambda z: z[1][axis])[1][axis]
        maxx = max(entities, key=lambda z: z[1][axis])[1][axis]
        def keyfunc(z):
            coords = tuple(z[1])
            return (coords[axis], ) + tuple(coords[:axis] + coords[axis+1:])
        s = sorted(entities, key=keyfunc, reverse=(dir==-1))

        divisions = numpy.linspace(minx, maxx, ndiv+1)
        (entities, coords) = zip(*s)
        coords = [c[axis] for c in coords]
        indices = numpy.searchsorted(coords[::dir], divisions)

        out = []
        for k in range(ndiv):
            out.append(entities[indices[k]:indices[k+1]])
        out.append(entities[indices[-1]:])

        return out

    def __call__(self, pc):
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        sentinel = object()
        sweeps = PETSc.Options(prefix).getString("pc_patch_construction_ps_sweeps", default=sentinel)
        if sweeps == sentinel:
            raise ValueError("Must set %spc_patch_construction_ps_sweeps" % prefix)

        out = []
        for sweep in sweeps.split(':'):
            axis = int(sweep[0])
            dir  = {'+': +1, '-': -1}[sweep[1]]
            ndiv = int(sweep[2:])

            entities = self.sort_entities(dm, axis, dir, ndiv)
            for patch in entities:
                iset = PETSc.IS().createGeneral(patch, comm=PETSc.COMM_SELF)
                out.append(iset)

        return out

class OrderedRelaxation(object):
    def __init__(self):
        self.name = None

    @staticmethod
    def callback(dm, entity, nclosures):
        raise NotImplementedError

    @staticmethod
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    def __call__(self, pc):
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        sentinel = object()
        opts = PETSc.Options(prefix)
        name = self.name
        assert self.name is not None

        codim = opts.getInt("pc_patch_construction_%s_codim" % name, default=sentinel)
        if codim == sentinel:
            dim = opts.getInt("pc_patch_construction_%s_dim" % name, default=0)
            entities = range(*dm.getDepthStratum(dim))
        else:
            entities = range(*dm.getHeightStratum(dim))

        nclosure = opts.getInt("pc_patch_construction_%s_nclosures" % name, default=1)

        sortorders = opts.getString("pc_patch_construction_%s_sort_order" % name, default=sentinel)
        if sortorders == sentinel:
            raise ValueError("Must set %spc_patch_construction_%s_sort_order" % (prefix, name))
        if sortorders != "None":
            outecs = []
            for sortorder in sortorders.split("|"):
                sortdata = []
                for axis in sortorder.split(':'):
                    ax = int(axis[0])
                    if len(axis) > 1:
                        sgn = {'+': 1, '-': -1}[axis[1]]
                    else:
                        sgn = 1
                    sortdata.append((ax, sgn))

                def keyfunc(z):
                    return tuple(sgn*z[1][ax] for (ax, sgn) in sortdata)

                ec_ = ((p, self.coords(dm, p)) for p in entities)
                outecs = chain(outecs, sorted(ec_, key=keyfunc))
            ec = outecs
        else:
            ec = ((p, self.coords(dm, p)) for p in entities)

        out = []
        for x in ec:
            subentities = self.callback(dm, x[0], nclosure)
            iset = PETSc.IS().createGeneral(subentities, comm=PETSc.COMM_SELF)
            out.append(iset)
        return out

class OrderedVanka(OrderedRelaxation):
    def __init__(self):
        self.name = "ovanka"

    @staticmethod
    def callback(dm, entity, nclosures):
        out = set()
        out.add(entity)

        for c in range(nclosures):
            new = set()
            for entity in out:
                star = dm.getTransitiveClosure(entity, useCone=False)[0]
                for p in star:
                    if p in out: continue
                    closure = dm.getTransitiveClosure(p, useCone=True)[0]
                    new.update(closure)
            out.update(new)
        return list(out)

class OrderedStar(OrderedRelaxation):
    def __init__(self):
        self.name = "ostar"

    @staticmethod
    def callback(dm, entity, nclosures):
        out = set()
        out.add(entity)

        for c in range(nclosures):
            star = dm.getTransitiveClosure(entity, useCone=False)[0]
            out.update(star)
        return list(out)

