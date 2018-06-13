import numpy
from functools import partial
from firedrake.petsc import PETSc

__all__ = ('PlaneSmoother', 'OrderedVanka', 'OrderedStar')


def select_entity(p, dm=None, exclude=None):
    """Filter entities based on some label.

    :arg p: the entity.
    :arg dm: the DMPlex object to query for labels.
    :arg exclude: The label marking points to exclude."""
    if exclude is None:
        return True
    else:
        # If the exclude label marks this point (the value is not -1),
        # we don't want it.
        return dm.getLabelValue(exclude, p) == -1


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
        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = [(p, self.coords(dm, p)) for p in
                    filter(select, range(*dm.getChart()))]

        minx = min(entities, key=lambda z: z[1][axis])[1][axis]
        maxx = max(entities, key=lambda z: z[1][axis])[1][axis]

        def keyfunc(z):
            coords = tuple(z[1])
            return (coords[axis], ) + tuple(coords[:axis] + coords[axis+1:])

        s = sorted(entities, key=keyfunc, reverse=(dir == -1))

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

        patches = []
        for sweep in sweeps.split(':'):
            axis = int(sweep[0])
            dir = {'+': +1, '-': -1}[sweep[1]]
            ndiv = int(sweep[2:])

            entities = self.sort_entities(dm, axis, dir, ndiv)
            for patch in entities:
                iset = PETSc.IS().createGeneral(patch, comm=PETSc.COMM_SELF)
                patches.append(iset)

        iterationSet = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
        return (patches, iterationSet)


class OrderedRelaxation(object):
    def __init__(self):
        self.name = None

    def callback(self, dm, entity, nclosures):
        raise NotImplementedError

    def set_options(self, dm, opts, name):
        pass

    @staticmethod
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    @staticmethod
    def get_entities(opts, name, dm):
        sentinel = object()
        codim = opts.getInt("pc_patch_construction_%s_codim" % name, default=sentinel)
        if codim == sentinel:
            dim = opts.getInt("pc_patch_construction_%s_dim" % name, default=0)
            entities = range(*dm.getDepthStratum(dim))
        else:
            entities = range(*dm.getHeightStratum(codim))
        return entities

    def __call__(self, pc):
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        sentinel = object()
        opts = PETSc.Options(prefix)
        name = self.name
        assert self.name is not None

        self.set_options(dm, opts, name)

        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = list(filter(select, self.get_entities(opts, name, dm)))

        nclosure = opts.getInt("pc_patch_construction_%s_nclosures" % name, default=1)
        patches = []
        for entity in entities:
            subentities = self.callback(dm, entity, nclosure)
            iset = PETSc.IS().createGeneral(subentities, comm=PETSc.COMM_SELF)
            patches.append(iset)

        # Now make the iteration set.
        iterset = []

        sortorders = opts.getString("pc_patch_construction_%s_sort_order" % name, default=sentinel)
        if sortorders == sentinel:
            sortorders = "None"

        if sortorders == "None":
            piterset = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
            return (patches, piterset)

        coords = list(enumerate(self.coords(dm, p) for p in entities))
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

            iterset += [x[0] for x in sorted(coords, key=keyfunc)]

        piterset = PETSc.IS().createGeneral(iterset, comm=PETSc.COMM_SELF)
        return (patches, piterset)


class OrderedVanka(OrderedRelaxation):
    def __init__(self):
        self.name = "ovanka"

    def set_options(self, dm, opts, name):
        sentinel = object()
        exclude_dim = opts.getInt("pc_patch_construction_%s_exclude_dim" % name, default=sentinel)
        if exclude_dim == sentinel:
            self.exclude_dim = None
        else:
            self.exclude_dim = exclude_dim
            self.exclude_range = dm.getDepthStratum(exclude_dim)

    def callback(self, dm, entity, nclosures):
        out = set()
        out.add(entity)

        if self.exclude_dim is None:
            for c in range(nclosures):
                new = set()
                for entity in out:
                    star = dm.getTransitiveClosure(entity, useCone=False)[0]
                    for p in star:
                        closure = dm.getTransitiveClosure(p, useCone=True)[0]
                        new.update(closure)
                out.update(new)
        else:
            for c in range(nclosures):
                new = set()
                for entity in out:
                    star = dm.getTransitiveClosure(entity, useCone=False)[0]
                    for p in star:
                        closure = dm.getTransitiveClosure(p, useCone=True)[0]
                        for x in closure:
                            if not (self.exclude_range[0] <= x < self.exclude_range[1]):
                                new.add(x)
                out.update(new)
        return list(out)


class OrderedStar(OrderedRelaxation):
    def __init__(self):
        self.name = "ostar"

    def callback(self, dm, entity, nclosures):
        out = set()
        out.add(entity)

        candidates = set([entity])

        while candidates:
            e = candidates.pop()
            star, _ = dm.getTransitiveClosure(e, useCone=False)
            nclosures -= 1
            if nclosures > 0:
                candidates.update(star)
            out.update(star)
        return list(out)
