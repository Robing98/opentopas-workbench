"""Beam model for the spot-by-spot recalculation: per nominal energy the phase space at the source plane.

The source sits upstream of the ripple filter at `source_distance` from the isocenter, as in spot_dist (115 cm).
Per energy: mean energy, energy spread [% of the mean], sigma, divergence and correlation in x and y (TOPAS emittance
source, BiGaussian), and particles per MU. Values between fitted energies are interpolated linearly in the nominal
energy; outside the fitted range the model refuses instead of extrapolating.

File format (JSON): {"name", "particle", "source_distance_mm", "vsad_x_mm", "vsad_y_mm", "dose_scale", "sources",
"energies": [{"nominal", "mean", "spread_pct", "sigma_x", "sigma_y", "div_x", "div_y", "corr_x", "corr_y",
"particles_per_mu"}, ...]} with sigma in mm and divergence in mrad.
"""
import bisect, json, math

KEYS = ("mean", "spread_pct", "sigma_x", "sigma_y", "div_x", "div_y", "corr_x", "corr_y", "particles_per_mu")


class BeamModel:
    def __init__(self, d):
        self.d = d
        self.name = d.get("name", "beam model")
        self.particle = d.get("particle", "proton")
        self.source_distance = float(d.get("source_distance_mm", 1150.0))
        self.vsad = (float(d.get("vsad_x_mm", 7549.0)), float(d.get("vsad_y_mm", 8219.0)))
        self.dose_scale = float(d.get("dose_scale", 1.0))   # absolute dose: Gy per (TOPAS Gy per particle * particles)
        rows = sorted(d["energies"], key=lambda r: r["nominal"])
        if not rows: raise ValueError("beam model without energies")
        for r in rows:
            for k in KEYS:
                if k not in r: raise ValueError(f"beam model, energy {r.get('nominal')}: missing {k}")
            if not -1 <= r["corr_x"] <= 1 or not -1 <= r["corr_y"] <= 1: raise ValueError("correlation outside [-1, 1]")
        self.rows = rows
        self.e = [float(r["nominal"]) for r in rows]

    @classmethod
    def load(cls, path):
        with open(path) as fh: return cls(json.load(fh))

    def save(self, path):
        with open(path, "w") as fh: json.dump(self.d, fh, indent=1)

    def at(self, energy, tol=1e-6):
        """Interpolated parameters for a nominal energy [MeV]."""
        e = float(energy)
        if e < self.e[0] - tol or e > self.e[-1] + tol:
            raise ValueError(f"energy {e:g} MeV outside the beam model ({self.e[0]:g} to {self.e[-1]:g} MeV)")
        i = bisect.bisect_left(self.e, e)
        if i < len(self.e) and abs(self.e[i] - e) <= tol: return {k: float(self.rows[i][k]) for k in KEYS}
        lo, hi = self.rows[i - 1], self.rows[i]
        t = (e - lo["nominal"]) / (hi["nominal"] - lo["nominal"])
        return {k: float(lo[k]) + t * (float(hi[k]) - float(lo[k])) for k in KEYS}

    def spot_at_source(self, x_iso, y_iso):
        """Position at the source plane [mm] and direction angles [rad] of a spot aimed at (x_iso, y_iso) at the isocenter.

        Each direction scans from its own virtual source (VSAD x and y), so the ray through the source plane points at the
        spot position: position = iso * (VSAD - zs) / VSAD, angle = atan(iso / VSAD).
        """
        zs = self.source_distance
        (vx, vy) = self.vsad
        return (x_iso * (vx - zs) / vx, y_iso * (vy - zs) / vy, math.atan2(x_iso, vx), math.atan2(y_iso, vy))


def generic(energies=None, sigma=2.97, div=1.0, spread=0.2, particles_per_mu=1.0e8):
    """The generic model of spot_dist: source 115 cm, sigma 2.97 mm, 1 mrad, spread 0.2 % (placeholders, not MIT data).

    Mean energy equals nominal energy. particles_per_mu is a placeholder: the plan's MU become particles only through the
    measured value of the machine. Use for public tests and to compare variants among themselves.
    """
    energies = energies or [60.0 + 5.0 * i for i in range(33)]   # 60 to 220 MeV
    rows = [{"nominal": e, "mean": e, "spread_pct": spread, "sigma_x": sigma, "sigma_y": sigma, "div_x": div, "div_y": div,
             "corr_x": 0.0, "corr_y": 0.0, "particles_per_mu": particles_per_mu} for e in energies]
    return BeamModel({"name": "generic (spot_dist placeholders)", "particle": "proton", "source_distance_mm": 1150.0,
                      "vsad_x_mm": 7549.0, "vsad_y_mm": 8219.0, "dose_scale": 1.0, "energies": rows,
                      "sources": {"all": "spot_dist placeholders (Simulationskette par. 6), not fitted"}})
