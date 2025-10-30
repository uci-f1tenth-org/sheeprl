import numpy as np, csv

# Load centerline (x,z) points from CSV → Nx2 NumPy array
def load_centerline_csv(path):
    xs, zs = [], []
    with open(path) as f:
        for xz in csv.reader(f):
            x, z = map(float, xz)
            xs.append(x); zs.append(z)
    return np.column_stack([np.array(xs), np.array(zs)])  # XZ -> XY plane


# Represents the track geometry: arc length, heading, tangent, normal, curvature
class FrenetPath:
    def __init__(self, xy, loop=True):
        self.xy = np.asarray(xy, float)
        self.loop = loop

        nxt = np.roll(self.xy, -1, axis=0)           # next waypoint
        dxy = nxt - self.xy                          # segment vectors
        seg_len = np.linalg.norm(dxy, axis=1)        # segment lengths
        if not loop: seg_len[-1] = 0.0

        self.s_wp = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])  # cumulative distance
        self.L = float(self.s_wp[-1] + seg_len[-1])                   # total length

        seg_psi = np.arctan2(dxy[:,1], dxy[:,0])     # heading along path
        self.psi_r = seg_psi
        self.t = np.column_stack((np.cos(seg_psi), np.sin(seg_psi)))  # tangents
        self.n = np.column_stack((-self.t[:,1], self.t[:,0]))         # normals

        # curvature κ = dψ/ds
        psi_f, psi_b = np.roll(seg_psi, -1), np.roll(seg_psi, 1)
        dpsi = psi_f - psi_b
        ds = np.roll(self.s_wp, -1) - np.roll(self.s_wp, 1)
        if not loop:
            dpsi[0]=dpsi[1]; dpsi[-1]=dpsi[-2]
            ds[0]=ds[1]; ds[-1]=ds[-2]
        self.kappa = dpsi / np.where(np.abs(ds) < 1e-6, 1e-6, ds)

    # Wrap or clamp s along the path length
    def wrap_s(self, s):
        return (s % self.L) if self.loop else np.clip(s, 0, self.L)


# Projects global (x,y,ψ,vx,vy) → Frenet (s,d,Δψ,vs,vd,κ)
class FrenetProjector:
    def __init__(self, P: FrenetPath):
        self.P = P

    def _nearest_idx(self, p):
        return int(np.argmin(np.sum((self.P.xy - p)**2, axis=1)))     # closest waypoint

    def project(self, x, y, psi, vx, vy):
        p = np.array([x, y], float)
        i = self._nearest_idx(p)
        j = (i + 1) % self.P.xy.shape[0]
        Pi, Pj = self.P.xy[i], self.P.xy[j]

        # Project point onto nearest path segment
        e = Pj - Pi; L2 = np.dot(e, e) + 1e-12
        u = np.clip(np.dot(p - Pi, e) / L2, 0.0, 1.0)
        proj = Pi + u * e

        # Longitudinal position (s)
        s = self.P.wrap_s(self.P.s_wp[i] + u * np.linalg.norm(e))
        t, n = self.P.t[i], self.P.n[i]
        psi_r, kappa_r = self.P.psi_r[i], self.P.kappa[i]

        # Lateral offset (d) and heading error (Δψ)
        d = np.dot(p - proj, n)
        dpsi = np.arctan2(np.sin(psi - psi_r), np.cos(psi - psi_r))

        # Velocity components in Frenet frame
        v = np.array([vx, vy], float)
        vs = np.dot(v, t) / max(1.0 - kappa_r * d, 1e-3)
        vd = np.dot(v, n)

        return dict(s=s, d=d, dpsi=dpsi, vs=vs, vd=vd,
                    kappa_r=kappa_r, psi_r=psi_r)
