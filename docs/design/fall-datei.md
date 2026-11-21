# Falldatei: Workbench, Lieferzeit-Simulator und MRT-Viewer an einem Fall

Entwurf, Stand 2026-11-21. Noch kein Code. Nur Forschung, keine Diagnose, kein Medizinprodukt.

**Kern.** Drei Werkzeuge bleiben getrennt und lesen und schreiben eine gemeinsame Datei `fall.json` in einem
Fallordner, den der Nutzer lokal wählt. Ein Fall hat ein Bezugssystem: das Raster des Planungs-CTs. Jede
Bildserie trägt ihre Transformation dorthin und deren Güte. Jedes Werkzeug schreibt nur seinen eigenen Abschnitt.
Ein Prüfskript sagt, ob die Datei stimmt und welche Ergebnisse veraltet sind.

- **Workbench** (dieses Repo): TOPAS-Dosis und LETd einer Strahlkonfiguration, mit oder ohne Rippelfilter (RiFi).
- **Lieferzeit-Simulator** (Repo `synchrotron-delivery-simulation`): Bestrahlungszeit und Lagen- und
  Spill-Struktur desselben Plans, mit und ohne RiFi.
- **MRT-Viewer** (statische HTML-Seite aus einer Vorlage und einem Python-Generator): Dosis, LETd und
  Ventrikelsaum auf CT, MRT und PET über den Verlauf.

## 1. Bestand

| Werkzeug | liest | schreibt |
|---|---|---|
| `tools/plan_mc` (Branch `otpas-2351-plan-mc-side-project`, noch nicht in main) | RT Ion Plan oder matRad-Plan, CT (DICOM), HLUT, Strahlmodell (JSON), RiFi-Variante | Laufordner: `letd.npy`, `rbe_dose_*.npy` [z, y, x] je Fraktion, `rbe_dose_1_1.dcm` (RT Dose, trägt das Raster) |
| `tools/mrt_overlay export` | Laufordner von plan_mc | `ct_frame/*.nii.gz` (LETd, Dosis physikalisch und RBE-gewichtet) und `export.json` |
| `tools/mrt_overlay viewer` | `cacheF` des Viewers (`log.json`, `ref_t1.nii.gz`, ein `*.npy` je Serie und Tag), CT, `ct_frame`, RT Structure Set, SynthSeg-Labels | Ebenen auf dem Viewer-Raster, `mri_to_ct.tfm`, `overlay.json` (Registriergüte je Untersuchung, Hinschau-Liste); in den Datenordner des Viewers `letd_all.js`, `marks_all.js`, `letd.json` |
| RiFi-Designer (PR #10, `generators/_rifi.py`) | nichts außer der Eingabe im Web UI | Filterdefinition `rifi/v1` (JSON: Lagen mit Periode, Profil oder Stufen, Material, SPR), Kern, TOPAS-Block |
| Simulator, `POST /plans`, `POST /plans/{id}/timing` | RT Ion Plan (Upload), Teilchen, Kalibrierung, optional ein Zyklus-Datensatz | `total_time_s` und eine freie `breakdown` je Feld und Lage |
| Simulator, `compute/rifi_model.py` | Energiebereich und Lagenzahl je Plan aus dem Zyklusexport, RiFi-Dicke, Designfaktor `kappa`, Lochanteil | Lagenzahl mit RiFi, Füllungen des Beschleunigers |
| Viewer, Generator | `cacheF` (alle Serien starr auf eine Referenz-T1 ausgerichtet, 8 Bit), Befundtexte, Ereignisse, Laborwerte, Ventrikelvolumina | eine HTML-Seite und je Serie und Tag ein Volumen als Kachelbild in einer `.js`-Datei (`MRT.vol(...)`), die LETd-Ebene nur wenn alle drei Dateien da sind |

Zwei Befunde aus dem Bestand prägen den Entwurf (beide gelesen im Code, also wahrscheinlich, nicht geprüft):

1. Der Viewer zeigt auf dem Raster seiner Referenz-T1, nicht auf dem CT-Raster. `mrt_overlay` rechnet im CT und
   bildet dann mit `mri_to_ct.tfm` auf das Viewer-Raster ab. Die Falldatei führt deshalb beide: das CT als
   Bezug, das Viewer-Raster als abgeleitete Sicht mit ihrer Transformation.
2. Der Simulator kennt einen einzelnen Plan (`/plans`) und das RiFi-Modell getrennt. Das RiFi-Modell rechnet
   auf Aggregaten des Zyklusexports. Für "derselbe Plan mit RiFi" fehlt eine Brücke (Folgeaufgabe 4).

## 2. Die Falldatei

**Format: JSON**, Schema `fall/v1`, wie `rifi/v1` und `studie/v1`. Gründe: der Workbench-Server läuft auf der
Standardbibliothek allein, und Python schreibt TOML nicht ohne Zusatzpaket; JSON liest auch die Viewer-Seite direkt.

**Pfade** stehen relativ zum Fallordner, mit `/`, ohne `..`, nie absolut. So zieht ein Fall um, ohne dass sich
die Datei ändert, und kein Pfad eines Rechners gerät in ein Ergebnis.

```json
{
 "schema": "fall/v1", "rev": 7, "fall_id": "demo-tg119",
 "bezug": {"serie": "ct_plan", "frame_of_reference_uid": "1.2.3", "raster": {"size": [256, 256, 160], "spacing_mm": [1.0, 1.0, 2.0]}},
 "plan": {"datei": "plan/rtplan.dcm", "sha256": "…", "teilchen": "proton", "fraktionen": 30, "verschreibung_gy_rbe": 60.0},
 "strukturen": {"datei": "plan/rtstruct.dcm", "sha256": "…", "ptv": "PTV"},
 "rifi": {"definition": "rifi/t4.json", "sha256": "…", "position_cm": 80.0},
 "maschine": {"strahlmodell": "modell/beam_model.json", "sha256": "…", "hlut": "modell/hlut.json"},
 "lieferung": {"kalibrierung": null, "datensatz": null, "spill": {"min_extraktion_s": null}},
 "bilder": [
  {"serie": "ct_plan", "modalitaet": "CT", "datum": "2001-01-01", "ordner": "bilder/ct_plan",
   "nach_bezug": null, "guete": null},
  {"serie": "t1_2001-02-01", "modalitaet": "MR", "art": "t1", "datum": "2001-02-01", "ordner": "bilder/t1_2001-02-01",
   "nach_bezug": "transform/t1_2001-02-01.tfm", "guete": {"verfahren": "mattes_mi", "rms_mm": 0.9, "ok": true}}
 ],
 "zeitleiste": [{"datum": "2001-03-01", "art": "biopsie", "befund": "texte/biopsie.txt", "ort": null}],
 "ergebnisse": {
  "workbench": [{"id": "mc-ohne-rifi", "rifi": false, "ordner": "ergebnisse/mc-ohne-rifi", "version": "git:8ae74af",
                 "eingaben": {"plan": "…", "maschine": "…"}, "dateien": {"letd.npy": "…", "rbe_dose_1_1.dcm": "…"}}],
  "simulator": [{"id": "zeit-ohne-rifi", "rifi": false, "version": "1.4.0", "eingaben": {"plan": "…"},
                 "gesamtzeit_s": 312.0, "lagen": 48, "datei": "ergebnisse/zeit-ohne-rifi.json"}],
  "viewer": [{"id": "verlauf", "aus": ["mc-ohne-rifi"], "ordner": "viewer/Verlauf-Daten", "version": "…"}]
 }
}
```

Alle Zahlen oben sind Platzhalter. `…` steht für einen SHA-256.

**Wer schreibt was.**

| Abschnitt | schreibt | liest |
|---|---|---|
| `fall_id`, `bezug`, `plan`, `strukturen`, `bilder` (ohne `nach_bezug`, `guete`), `zeitleiste` | Nutzer, Hilfe: ein Scan-Befehl, der DICOM-Kopfdaten in Einträge übersetzt | alle |
| `rifi`, `maschine` | Workbench (RiFi-Designer, Strahlmodell) | Workbench, Simulator |
| `lieferung`, `ergebnisse.simulator` | Simulator | Viewer (Zeit als Zahl in der Legende) |
| `ergebnisse.workbench` | Workbench (plan_mc, `mrt_overlay export`) | Viewer |
| `bilder[].nach_bezug`, `bilder[].guete`, `ergebnisse.viewer` | `mrt_overlay viewer` | Viewer |

**Versionierung und Schreiben.** `rev` zählt jede Änderung hoch. Ein Werkzeug liest die Datei, ändert nur seinen
Abschnitt und schreibt sie über eine temporäre Datei mit Umbenennen zurück; ist `rev` beim Schreiben ein anderes
als beim Lesen, liest es neu und wiederholt. Neue Felder sind erlaubt und ändern das Schema nicht; ein Feld mit
neuer Bedeutung heißt `fall/v2` und das Prüfskript bringt eine Umstellung von v1 mit.

**Ergebnisse mit Herkunft.** Jedes Ergebnis nennt die Version des Werkzeugs (Git-Commit oder Release), den Hash
jeder Eingabe, die es verbraucht hat, und den Hash jeder Datei, die es geschrieben hat. Ändert sich eine Eingabe,
ist das Ergebnis veraltet, ohne dass jemand das von Hand vermerken muss.

**Prüfskript** `fall_check.py FALLORDNER` (nur Standardbibliothek): prüft das Schema, dass jeder Pfad relativ ist
und im Fallordner liegt, dass jede genannte Datei da ist und ihr Hash stimmt, dass jede Serie außer dem Bezug eine
Transformation hat, und meldet je Ergebnis `aktuell`, `veraltet` (Eingabe geändert) oder `kaputt` (Datei fehlt).
Es endet mit rc 0 nur ohne `kaputt`. Testfälle: ein synthetischer Fall mit 1 und mit 3 Serien.

**Datenzonen.** Ein Fall liegt ganz in einer Zone. Ein Demo-Fall aus öffentlichen Daten (TG119 und das
synthetische MRT-Phantom aus `mrt_overlay/phantom.py`) liegt unter `workspace/offen/` und dient als Testfall. Ein
Fall, dessen Dosis aus einem klinischen Plan, einer klinischen HLUT oder Maschinendaten einer Anlage stammt, liegt
in der lokalen Zone, und mit ihm jedes Ergebnis daraus. Die Werkzeuge erzwingen das nicht; das Prüfskript warnt,
wenn ein Fallordner unter `workspace/offen/` einen RT Plan ohne Kennzeichen `"oeffentlich": true` nennt.

**Simulator lokal mit einem eigenen Plan.** Der Simulator läuft auf dem Rechner des Nutzers. Ein Befehl
`fall run FALLORDNER` im Simulator lädt `plan.datei` wie ein Upload, rechnet die Zeit ohne RiFi und mit der
RiFi-Konfiguration aus `rifi` und schreibt `ergebnisse.simulator`. Kein Plan und kein Wert verlässt den Rechner.
Für Tests nimmt er nur synthetische Zyklusdaten und Phantom-Pläne.

## 3. Viewer: CT und PET

**CT.** Die Volumen des Viewers sind 8 Bit, ein Fenster wäre damit fest. Vorschlag: HU + 1024 als 16 Bit,
verlustfrei in zwei Kanälen eines PNG-Kachelbilds (hohes und niedriges Byte), die Seite setzt den Wert beim Laden
zusammen und fenstert live. Voreinstellungen: Hirn, Weichteil, Knochen, frei wählbar mit Mitte und Breite. Die
verlustbehaftete WebP-Kodierung, die der Viewer für die MRTs nutzt, scheidet hier aus: sie verfälscht die Bytes. Das CT kommt mit der Umkehr von `mri_to_ct.tfm` auf
das Viewer-Raster, die `mrt_overlay viewer` schon berechnet.

**PET, Weg A: SUV (körpergewichtsbezogen).** SUV = C · Gewicht / Aktivität zur Aufnahmezeit. Benötigte Felder:

- `Units` (0054,1001) muss `BQML` sein; sonst ist die Serie schon umgerechnet oder nicht quantitativ.
- `DecayCorrection` (0054,1102): `START` heißt, die Voxel sind auf den Beginn der Serie korrigiert; dann wird die
  Aktivität von der Injektion bis dahin abgeklungen: A = A0 · 2^(−Δt / T½). `ADMIN` heißt, keine Abklingrechnung.
- `PatientWeight` (0010,1030) in kg.
- `RadiopharmaceuticalInformationSequence` (0054,0016) mit `RadionuclideTotalDose` (0018,1074, Bq),
  `RadionuclideHalfLife` (0018,1075, s) und `RadiopharmaceuticalStartDateTime` (0018,1078) oder
  `RadiopharmaceuticalStartTime` (0018,1072).
- Startzeit der Aufnahme: `SeriesDate`/`SeriesTime` (0008,0021/0031); weil einige Geräte sie bei einer
  Nachrekonstruktion neu setzen, die früheste `AcquisitionDateTime` oder `AcquisitionTime` der Schichten zum Abgleich.
- `RescaleSlope` und `RescaleIntercept` je Schicht, nicht je Serie: PET-Schichten haben oft eigene Faktoren.

Fehlt ein Feld, zeigt der Viewer die Serie nicht als SUV, sondern sagt, welches Feld fehlt, und bietet Weg B an.

**PET, Weg B: Tumor-zu-Hintergrund-Verhältnis (TBR).** Voxel geteilt durch den Mittelwert einer
Hintergrundregion. Braucht weder Gewicht noch Dosis und ist zwischen Geräten robuster. Der Hintergrund hängt vom
Tracer ab: bei einem Aminosäure-Tracer üblich gesundes Hirngewebe der Gegenseite, bei FDG ist die Hirnrinde selbst
stark speichernd und ein TBR schwerer zu deuten. Die Region legt der Nutzer fest; sie steht mit Serie und Maske
in der Falldatei, damit der Wert nachvollziehbar bleibt. Schwellen für Tumorvolumina übernimmt der Viewer nicht
selbst, sondern nur aus einer genannten Leitlinie, mit Quelle in der Legende.

**Fusion.** PET direkt starr auf das Planungs-CT ist mit Mutual Information schwach, weil PET grob aufgelöst ist
und wenig Anatomie zeigt. Besser über das Begleitbild derselben Aufnahme: bei PET/CT liegt das Niedrigdosis-CT
im selben Bezugssystem (gleiche `FrameOfReferenceUID`), es wird mit `register.py` auf das Planungs-CT registriert
(CT auf CT) und die Transformation auf das PET übertragen; bei PET/MR ebenso über das MR. Die Güte kommt wie bei
den MRTs in `bilder[].guete`. Anzeige als Farbebene über dem Grauwertbild, mit Skala in der JSON wie `let_scale`.

## 4. Biopsie

Nur als möglicher Eintrag in `zeitleiste`, ohne Annahme zu einem Befund: `art: "biopsie"`, Datum, optional ein
Verweis auf einen Befundtext im Fallordner (der Viewer zeigt ihn als Text, er wertet ihn nicht aus), optional der
Entnahmeort als Punkt oder Maske auf einer genannten MR-Serie (`ort: {"serie": …, "voxel_xyz": […]}`). Schnitt-Scans
der Pathologie wären später eine eigene Modalität (`modalitaet: "SM"`, Whole Slide Imaging) ohne Registrierung auf
das MRT; der Eintrag verweist dann nur auf sie.

## 5. Reihenfolge der Umsetzung

| # | Aufgabe | Größe | Nutzen |
|---|---|---|---|
| 1 | `fall/v1` als Schema, `fall_check.py`, Demo-Fall in `workspace/offen/` | S | Alle Werkzeuge haben ein Ziel, gegen das sie testen. |
| 2 | `mrt_overlay` liest Pfade aus `fall.json` statt zehn Argumenten, schreibt `nach_bezug`, `guete`, `ergebnisse.viewer` | M | Ein Start statt zehn Pfadabfragen; Registriergüte bleibt beim Fall. |
| 3 | plan_mc schreibt `ergebnisse.workbench` mit Version und Hashes, RiFi-Designer schreibt `rifi` | S | Jede Dosis weiß, aus welchem Plan und Filter sie stammt. |
| 4 | Simulator: Brücke Einzelplan mit RiFi (`rifi_model` auf einen geparsten Plan statt Zyklus-Aggregate), `fall run` | M | Zeit mit und ohne RiFi für genau den Plan, den die Workbench rechnet. |
| 5 | Viewer: CT-Ebene mit 16-Bit-Kacheln und HU-Fenstern | M | Dosis und LETd auf der Anatomie, aus der sie gerechnet sind. |
| 6 | Viewer: PET, SUV und TBR, Fusion über das Begleit-CT oder -MR | L | Stoffwechsel neben Dosis, LETd und Saum im Verlauf. |
| 7 | Zeitleiste: Biopsie-Eintrag mit Befundtext und Entnahmeort | S | Der Verlauf zeigt, wann und wo Gewebe entnommen wurde. |
