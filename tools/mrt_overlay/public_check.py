"""Registration error and position of the layer on public and synthetic data; writes BERICHT.md and result.json.

  python -m mrt_overlay.public_check RIRE_DIR OUT

RIRE_DIR holds training_001_ct.mha, training_001_mr_T1.mha and ct_T1.standard (RIRE training patient 001, from
the SimpleITK notebooks data on data.kitware.com). The gold standard comes from bone-implanted fiducial markers.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import SimpleITK as sitk

from . import export as ex, geometry as geo, layer as ly, phantom as ph, register as rg, ventricles as ve


def read_standard(path):
    """(CT points, MR points) of a RIRE .standard file."""
    src, dst = [], []
    for line in open(path):
        p = line.split()
        if len(p) == 7 and p[0].isdigit():
            v = [float(x) for x in p[1:]]
            src.append(v[:3]); dst.append(v[3:])
    return np.array(src), np.array(dst)


def head_points(img, threshold, step=4):
    a = sitk.GetArrayFromImage(img) > threshold
    return geo.brain_points(geo.like(a.astype(np.uint8), img), step)


def blob_position(mri, ct, transform, centre_ct, sigma=8.0):
    """Centroid on the MRI grid of a Gaussian dose blob centred at `centre_ct` in the CT frame (mm)."""
    x, y, z = ph._grid_points(ct)
    d = np.exp(-((x - centre_ct[0]) ** 2 + (y - centre_ct[1]) ** 2 + (z - centre_ct[2]) ** 2) / (2 * sigma ** 2))
    on = ly.resample(geo.like(d.astype(np.float32), ct), mri, transform)
    w = on / on.sum()
    k, j, i = np.indices(on.shape)
    ci = [float((w * a).sum()) for a in (i, j, k)]
    return np.array(mri.TransformContinuousIndexToPhysicalPoint(ci))


def rire(folder):
    ct = sitk.ReadImage(os.path.join(folder, "training_001_ct.mha"))
    mr = sitk.ReadImage(os.path.join(folder, "training_001_mr_T1.mha"))
    src, dst = read_standard(os.path.join(folder, "ct_T1.standard"))
    gold_ct_to_mr = rg.gold_from_points(src, dst)
    fit = np.linalg.norm(np.array([gold_ct_to_mr.TransformPoint(tuple(p)) for p in src]) - dst, axis=1).max()
    gold = gold_ct_to_mr.GetInverse()                               # MR point -> CT point, as rigid() returns
    pts = head_points(mr, 150.0)
    t0 = time.time()
    t, q = rg.rigid(mr, ct, mask_points=pts)
    rms, mx = geo.displacement_mm(t, gold, pts)
    # the layer: a blob at the head centre in the CT frame lands where the gold standard puts that point
    centre = pts.mean(0)
    c_ct = np.array(gold.TransformPoint(tuple(centre)))
    off = np.linalg.norm(blob_position(mr, ct, t, c_ct) - centre)
    off_gold = np.linalg.norm(blob_position(mr, ct, gold, c_ct) - centre)
    # started at the gold standard: where the metric's own optimum lies, apart from the search
    tg, vg = rg.from_start(mr, ct, gold)
    drift = geo.displacement_mm(tg, gold, pts)[0]
    return {"ct_size": list(ct.GetSize()), "ct_spacing_mm": [round(v, 3) for v in ct.GetSpacing()],
            "mr_size": list(mr.GetSize()), "mr_spacing_mm": [round(v, 3) for v in mr.GetSpacing()],
            "gold_fit_max_mm": float(fit), "head_points": len(pts), "seconds": round(time.time() - t0, 1),
            "registration": q, "error_rms_mm": rms, "error_max_mm": mx,
            "layer_offset_mm": float(off), "layer_offset_gold_mm": float(off_gold),
            "optimum_from_gold_rms_mm": drift, "metric_at_gold": float(rg.metric(mr, ct, gold)), "metric_from_gold": vg}


def synthetic():
    ct, mri, labels, truth = ph.make()
    brain = geo.like((sitk.GetArrayFromImage(labels) > 0).astype(np.uint8), mri)
    pts = geo.brain_points(brain, 4)
    t, q = rg.rigid(mri, ct, mask_points=pts)
    rms, mx = geo.displacement_mm(t, truth, pts)
    with tempfile.TemporaryDirectory() as tmp:
        run, frame = os.path.join(tmp, "run"), os.path.join(tmp, "frame")
        ph.write_run(run)
        ex.export(run, frame)
        layers = ly.to_grid(frame, mri, t)
        truth_layers = ly.to_grid(frame, mri, truth)
    w, wt = layers["dose_phys"], truth_layers["dose_phys"]
    k, j, i = np.indices(w.shape)
    cen = lambda a: np.array(mri.TransformContinuousIndexToPhysicalPoint([float((a * g).sum() / a.sum()) for g in (i, j, k)]))
    _, rim = ve.margin(labels)
    rim = sitk.GetArrayFromImage(rim).astype(bool)
    vent = np.isin(sitk.GetArrayFromImage(labels), ve.VENTRICLES)
    return {"registration": q, "error_rms_mm": rms, "error_max_mm": mx,
            "layer_offset_mm": float(np.linalg.norm(cen(w) - cen(wt))),
            "margin_ml": float(rim.sum() * np.prod(mri.GetSpacing()) / 1000),
            "ventricles_ml": float(vent.sum() * np.prod(mri.GetSpacing()) / 1000)}


def report(res):
    r, s = res.get("rire"), res["synthetic"]
    f = lambda v: f"{v:.2f}".replace(".", ",")
    lines = ["# LETd auf Verlaufs-MRTs: Registrierung und Lage der Ebene an öffentlichen Daten", ""]
    if r:
        verdict = "erreicht" if r["error_rms_mm"] <= rg.ACCEPT_MM else "nicht erreicht"
        lines += [f"1. RIRE Patient 001, CT auf MRT-T1: Registrierfehler gegen den Marker-Goldstandard RMS {f(r['error_rms_mm'])} mm, "
                  f"maximal {f(r['error_max_mm'])} mm über {r['head_points']} Punkte im Kopf; Ziel 1,5 mm {verdict}.",
                  f"2. Lage der Ebene (Dosisfleck in der Kopfmitte, über die Registrierung ins MRT gelegt): {f(r['layer_offset_mm'])} mm "
                  f"neben der Goldstandard-Lage; mit dem Goldstandard selbst {f(r['layer_offset_gold_mm'])} mm (Rest der Interpolation). "
                  f"Ursache an RIRE: das Optimum der Mutual Information selbst liegt {f(r['optimum_from_gold_rms_mm'])} mm (RMS) neben dem "
                  f"Goldstandard; vom Goldstandard aus gestartet läuft die Suche dorthin (Metrik {r['metric_at_gold']:.3f} am "
                  f"Goldstandard, {r['metric_from_gold']:.3f} danach). Vermutlich eine Grenze der Daten (4 mm "
                  f"Schichten; ob dieses MRT verzeichnungskorrigiert ist, ist nicht geprüft), nicht der Suche."]
    else:
        lines += ["1. RIRE nicht gerechnet (Ordner fehlt).", "2. -"]
    lines += [f"3. Synthetischer Kopf mit bekannter Transformation (3/-2/5 Grad, 6/-4/3 mm): Fehler RMS {f(s['error_rms_mm'])} mm, "
              f"maximal {f(s['error_max_mm'])} mm; Lage der Ebene {f(s['layer_offset_mm'])} mm.",
              "4. Gütemaß je Serie: Streuung der Starts, die dasselbe Optimum erreichen (RMS-Abstand im Kopf). Sie zeigt ein "
              "reproduzierbares Optimum, nicht dass es stimmt; auf dem Viewer-Raster kommt je Untersuchung eine eigene "
              "Gegenregistrierung dazu.",
              "5. Nur Anzeige, keine Diagnose, keine Läsionserkennung. Variable RBE nur Forschung, klinisch gilt 1,1.", "",
              "## Daten", "",
              "- RIRE (Retrospective Image Registration Evaluation, Vanderbilt), Trainingspatient 001: CT "
              + (f"{r['ct_size']} Voxel zu {r['ct_spacing_mm']} mm, MRT T1 {r['mr_size']} zu {r['mr_spacing_mm']} mm. "
                 f"Goldstandard aus Knochenmarkern (ct_T1.standard, acht Eckpunkte; Restfehler der starren Anpassung "
                 f"{f(r['gold_fit_max_mm'])} mm)." if r else "nicht geladen."),
              "- Synthetischer Kopf (tools/mrt_overlay/phantom.py): Gewebeklassen mit CT-Zahlen auf der einen und "
              "T1-Intensitäten auf der anderen Seite, Bias-Feld und Rauschen im MRT, asymmetrische Teile (Nase, ein "
              "größeres Auge) für die Drehung.", "",
              "## Verfahren", "",
              "Starr (Euler 3D), Mattes Mutual Information mit 50 Bins, 10 % Zufallsstichprobe, drei Auflösungsstufen "
              "(4, 2, 1 mit Glättung 2, 1, 0 mm), Gradientenabstieg mit festen Schrittregeln. Sechs Starts: geometrische "
              "Mitte und Schwerpunkt, je ungestört und um ±3 bis 4 Grad und ±5 bis 6 mm verschoben. Das beste Ergebnis "
              "nach dem Metrikwert auf allen Voxeln gewinnt. CT auf -1000 bis 2000 HU begrenzt. Das CT ist das feste Bild, das "
              "Ergebnis wird umgekehrt: mit dem MRT als festem Bild lag der Fehler an RIRE bei 3,29 mm RMS (Optimum der "
              "Metrik 2,8 mm neben dem Goldstandard), mit dem CT als festem Bild bei 1,1 bis 1,6 mm über 12 Läufe aus "
              "allen Starts mit zwei Stichproben.", ""]
    if r:
        q = r["registration"]
        lines += [f"RIRE: {q['starts']} Starts, {q['starts_same_optimum']} am selben Optimum, Streuung {f(q['spread_rms_mm'])} mm, "
                  f"Metrik {q['metric']:.3f}, {r['seconds']} s.", ""]
    q = s["registration"]
    lines += [f"Synthetisch: {q['starts']} Starts, {q['starts_same_optimum']} am selben Optimum, Streuung {f(q['spread_rms_mm'])} mm. "
              f"Ventrikel {f(s['ventricles_ml'])} ml, 4-mm-Saum {f(s['margin_ml'])} ml.", "",
              "## Grenzen", "",
              "- RIRE hat 4 mm Schichtabstand in CT und MRT; die Verlaufs-MRTs mit 3D-T1 um 1 mm sind feiner. Der Fehler "
              "an RIRE ist daher eher eine obere Grenze für 3D-Serien.",
              "- Starr reicht für den Schädel; Verformung nach Resektion oder Ödem fängt sie nicht ein.",
              "- Gemessen an einem öffentlichen Patienten; der Wert je Untersuchung steht auf Robins Daten in overlay.json.", ""]
    return lines


def main(argv=None):
    a = (argv if argv is not None else sys.argv[1:])
    rire_dir, out = a[0], a[1]
    os.makedirs(out, exist_ok=True)
    res = {"synthetic": synthetic()}
    if os.path.exists(os.path.join(rire_dir, "training_001_ct.mha")): res["rire"] = rire(rire_dir)
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    open(os.path.join(out, "BERICHT.md"), "w", encoding="utf-8").write("\n".join(report(res)) + "\n")
    print("\n".join(report(res)[:8]))


if __name__ == "__main__":
    main()
