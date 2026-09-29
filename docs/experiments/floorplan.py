"""Draw a top-down navmesh floor plan of a Matterport scene as SVG path data, cropped to a region of interest.
Needs habitat_sim.PathFinder only (no GPU / no simulator).  1 metre = 40 svg units."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELL = 0.1
SC = 4.0          # svg units per cell


class Floor:
    def __init__(self, scene, height=0.17, crop=None):
        """crop = (xmin, zmin, xmax, zmax) in metres; None = whole scene."""
        import habitat_sim
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(f"{REPO}/data/scene_datasets/mp3d/{scene}/{scene}.navmesh")
        self.top = pf.get_topdown_view(CELL, height)
        self.x0, self.z0 = float(pf.get_bounds()[0][0]), float(pf.get_bounds()[0][2])
        if crop is None:
            self.cx0, self.cz0, self.cx1, self.cz1 = self.x0, self.z0, self.x0 + self.top.shape[1] * CELL, self.z0 + self.top.shape[0] * CELL
        else:
            self.cx0, self.cz0, self.cx1, self.cz1 = crop
        self.w = (self.cx1 - self.cx0) / CELL * SC
        self.h = (self.cz1 - self.cz0) / CELL * SC

    def px(self, x):
        return (x - self.cx0) / CELL * SC

    def pz(self, z):
        return (z - self.cz0) / CELL * SC

    def path(self):
        """Row-run-length rectangles of the navigable cells inside the crop, as one SVG path string."""
        j0 = max(0, int((self.cx0 - self.x0) / CELL))
        j1 = min(self.top.shape[1], int((self.cx1 - self.x0) / CELL) + 1)
        i0 = max(0, int((self.cz0 - self.z0) / CELL))
        i1 = min(self.top.shape[0], int((self.cz1 - self.z0) / CELL) + 1)
        out = []
        for i in range(i0, i1):
            j = j0
            while j < j1:
                if self.top[i, j]:
                    s = j
                    while j < j1 and self.top[i, j]:
                        j += 1
                    x = (self.x0 + s * CELL - self.cx0) / CELL * SC
                    y = (self.z0 + i * CELL - self.cz0) / CELL * SC
                    out.append(f"M{x:.0f} {y:.0f}h{(j - s) * SC:.0f}v{SC:.0f}h-{(j - s) * SC:.0f}z")
                else:
                    j += 1
        return "".join(out)
