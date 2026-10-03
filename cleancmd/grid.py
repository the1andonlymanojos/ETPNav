"""The shared floor grid every method's answer is converted to before scoring.

Coordinates are Habitat world coordinates: x and z span the floor, y is up. The same scene file gives the same world
frame in habitat-sim 0.1.7 (ETPNav) and 0.3.1 (VLMaps), so cells are comparable across environments.
A cell is (row, col): row indexes z, col indexes x, cell (0, 0) has its lower corner at `origin`.
Plain numpy, Python 3.6+, so both conda envs and the scorer can import it.
"""
import math

import numpy as np


class Grid(object):
    def __init__(self, origin_xz, shape, cell=0.5, floor_y=None):
        self.x0, self.z0 = float(origin_xz[0]), float(origin_xz[1])
        self.rows, self.cols = int(shape[0]), int(shape[1])
        self.cell = float(cell)
        self.floor_y = None if floor_y is None else float(floor_y)   # which floor this grid covers (multi-floor houses)

    @classmethod
    def from_bounds(cls, xmin, zmin, xmax, zmax, cell=0.5, floor_y=None):
        """Grid covering a floor's bounding box, origin snapped down to a multiple of `cell` so it is reproducible."""
        x0 = math.floor(xmin / cell) * cell
        z0 = math.floor(zmin / cell) * cell
        cols = int(math.ceil((xmax - x0) / cell))
        rows = int(math.ceil((zmax - z0) / cell))
        return cls((x0, z0), (rows, cols), cell, floor_y)

    def to_dict(self):
        return {"origin_xz": [self.x0, self.z0], "shape": [self.rows, self.cols], "cell": self.cell, "floor_y": self.floor_y}

    @classmethod
    def from_dict(cls, d):
        return cls(d["origin_xz"], d["shape"], d.get("cell", 0.5), d.get("floor_y"))

    def __eq__(self, other):
        return isinstance(other, Grid) and self.to_dict() == other.to_dict()

    def to_cell(self, x, z):
        return int(math.floor((z - self.z0) / self.cell)), int(math.floor((x - self.x0) / self.cell))

    def center(self, r, c):
        return self.x0 + (c + 0.5) * self.cell, self.z0 + (r + 0.5) * self.cell

    def inside(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cells_within(self, x, z, radius):
        """Cells whose centre lies within `radius` metres of (x, z). Always includes the cell containing the point."""
        out = set()
        r0, c0 = self.to_cell(x, z)
        k = int(math.ceil(radius / self.cell)) + 1
        for r in range(r0 - k, r0 + k + 1):
            for c in range(c0 - k, c0 + k + 1):
                if not self.inside(r, c):
                    continue
                cx, cz = self.center(r, c)
                if (cx - x) ** 2 + (cz - z) ** 2 <= radius ** 2:
                    out.add((r, c))
        if self.inside(r0, c0):
            out.add((r0, c0))
        return out

    def mask(self, cells):
        m = np.zeros((self.rows, self.cols), dtype=bool)
        for r, c in cells:
            if self.inside(r, c):
                m[r, c] = True
        return m
