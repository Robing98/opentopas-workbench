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
        for j in (i, i - 1):   # within tol of a row, also just above the top energy where bisect gives len
            if 0 <= j < len(self.e) and abs(self.e[j] - e) <= tol: return {k: float(self.rows[j][k]) for k in KEYS}
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


FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def with_plan_spots(model, plan):
    """Copy of `model` whose sigma per energy reproduces the plan's ScanningSpotSize at the isocenter.

    DICOM states the spot size as FWHM in air at the isocenter; the model's sigma sits at the source plane. With the
    model's divergence d and correlation r kept, the drift over the source distance z gives
    sigma_iso^2 = s^2 + 2 r s d z + d^2 z^2, solved here for s. The drift is taken in vacuum: the air between source
    and isocenter that TOPAS simulates widens the spot a little more. The values are the clinic's beam data: they go
    into the run folder only and into no message.
    """
    z = model.source_distance
    rows = {}
    for f in plan.fields:
        for l in f.layers:
            if not l.spot_size: raise ValueError(f"field {f.name}: layer at {l.energy:g} MeV without ScanningSpotSize")
            base = model.at(l.energy)
            row = dict(base, nominal=float(l.energy))
            for k, fwhm in zip("xy", l.spot_size):
                s_iso, d, r = float(fwhm) / FWHM_PER_SIGMA, base[f"div_{k}"] / 1000.0, base[f"corr_{k}"]
                disc = s_iso ** 2 - (d * z) ** 2 * (1.0 - r * r)
                s = -r * d * z + math.sqrt(disc) if disc > 0 else -1.0
                if s <= 0:
                    raise ValueError(f"field {f.name}: the plan's spot size at {l.energy:g} MeV is smaller than the "
                                     f"model's divergence alone gives at the isocenter")
                row[f"sigma_{k}"] = s
            key = round(float(l.energy), 6)
            if key in rows and any(abs(rows[key][k] - row[k]) > 1e-3 for k in ("sigma_x", "sigma_y")):
                raise ValueError(f"layers at {l.energy:g} MeV state different spot sizes: run their fields on their own")
            rows[key] = row
    d = dict(model.d, name=model.name + ", spot sizes from the plan", energies=sorted(rows.values(), key=lambda r: r["nominal"]))
    d["sources"] = dict(model.d.get("sources", {}), sigma="ScanningSpotSize of the plan (FWHM / 2.3548), projected back "
                        "to the source plane with the model's divergence and correlation, drift in vacuum")
    return BeamModel(d)
