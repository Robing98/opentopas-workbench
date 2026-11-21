# Falldatei: Workbench, Lieferzeit-Simulator und MRT-Viewer an einem Fall

Entwurf, Stand 2026-11-21, nachgezogen auf Robins Entscheidungen vom selben Tag (PET, Biopsie, Umstieg des
Viewers). Noch kein Code. Nur Forschung, keine Diagnose, kein Medizinprodukt.

**Kern.** Drei Werkzeuge bleiben getrennt und lesen und schreiben eine gemeinsame Datei `fall.json` in einem
Fallordner, den der Nutzer lokal wählt. Ein Fall hat ein Bezugssystem: das Raster des Planungs-CTs. Jede
Bildserie trägt ihre Transformation dorthin und deren Güte. Jedes Werkzeug schreibt nur seinen eigenen Abschnitt.
Ein Prüfskript sagt, ob die Datei stimmt und welche Ergebnisse veraltet sind.

- **Workbench** (dieses Repo): TOPAS-Dosis und LETd einer Strahlkonfiguration, mit oder ohne Rippelfilter (RiFi).
- **Lieferzeit-Simulator** (Repo `synchrotron-delivery-simulation`): Bestrahlungszeit und Lagen- und
  Spill-Struktur desselben Plans, mit und ohne RiFi.
- **MRT-Viewer** (heute eine statische HTML-Seite aus einer Vorlage und einem Python-Generator, künftig eine App
  mit React und NiiVue, Abschnitt 6): Dosis, LETd und Ventrikelsaum auf CT, MRT und PET über den Verlauf.

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
 "zeitleiste": [{"datum": "2000-12-01", "art": "biopsie", "ausgangszustand": true, "befund": "texte/biopsie.txt", "ort": null}],
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
Fall aus eigenen Aufnahmen, nur mit öffentlichen Werkzeugen und öffentlichen Daten gerechnet, liegt unter
`workspace/eigen/`. Ein Fall, dessen Dosis aus einem klinischen Plan, einer klinischen HLUT oder Maschinendaten einer
Anlage stammt, liegt in der lokalen Zone, und mit ihm jedes Ergebnis daraus. Die Werkzeuge erzwingen das nicht; das Prüfskript warnt,
wenn ein Fallordner unter `workspace/offen/` einen RT Plan ohne Kennzeichen `"oeffentlich": true` nennt.

**Simulator lokal mit einem eigenen Plan.** Der Simulator läuft auf dem Rechner des Nutzers. Ein Befehl
`fall run FALLORDNER` im Simulator lädt `plan.datei` wie ein Upload, rechnet die Zeit ohne RiFi und mit der
RiFi-Konfiguration aus `rifi` und schreibt `ergebnisse.simulator`. Kein Plan und kein Wert verlässt den Rechner.
Für Tests nimmt er nur synthetische Zyklusdaten und Phantom-Pläne.

## 3. Viewer: CT und PET

**CT.** Das CT kommt mit der Umkehr von `mri_to_ct.tfm` auf das Viewer-Raster, die `mrt_overlay viewer` schon
berechnet, als NIfTI mit HU in 16 Bit (int16). Fenster: Hirn, Weichteil, Knochen, frei wählbar mit Mitte und
Breite. Die CT-Ebene wird gleich in der neuen App gebaut (Abschnitt 6), nicht mehr auf der alten Seite: NiiVue
fenstert ein int16-Volumen von sich aus, der frühere Plan mit HU + 1024 in zwei Kanälen eines PNG-Kachelbilds
entfällt. Die verlustbehaftete WebP-Kodierung der alten Seite schied für das CT ohnehin aus, weil sie die Werte
verfälscht.

**PET: Annahme des Entwurfs.** Ausgelegt ist der Viewer auf einen Aminosäure-Tracer. Das legt den Weg fest:
zuerst das Verhältnis zu einer Referenzregion, SUV nur als zweiter Weg mit von Hand eingetragenen Werten.

**Lücken im Kopf der PET-Datei, allgemein.** Eine PET-Serie, besonders ein Summenbild aus einer dynamischen
Aufnahme oder eine Serie eines älteren Scanners, kann folgende Lücken haben. Der Viewer und `pet.py` rechnen mit
jeder davon:

- `Units` (0054,1001) und `DecayCorrection` (0054,1102) können `NONE` sein: dann ist offen, ob die Voxel Bq/ml
  sind und worauf sie abklingkorrigiert sind.
- `PatientWeight` (0010,1030) und `RadionuclideTotalDose` (0018,1074) können 0 sein.
- `ImagePositionPatient` (0020,0032) kann bei 0,0,0 beginnen, mit Einheitsorientierung in
  `ImageOrientationPatient`: dann gibt es keinen verlässlichen Patientenrahmen, und die `FrameOfReferenceUID`
  verbindet die Serie mit nichts.
- `RescaleSlope` und `RescaleIntercept` stehen je Schicht, nicht je Serie.

Daraus folgen zwei Pflichten:

1. **Registrierung von Grund auf.** Die Lage aus dem Kopf wird nicht verwendet. `pet.py` liest das Volumen mit
   Ursprung 0 und registriert es starr (Euler3D, Mattes Mutual Information, mehrere Starts) auf die 3D-T1 des
   Viewer-Rasters. Gibt es ein Begleitbild derselben Aufnahme (Niedrigdosis-CT bei PET/CT, MR bei PET/MR) mit
   verlässlichem Kopf, ist der Weg darüber besser: Begleitbild auf das Planungs-CT oder die T1 registrieren und die
   Transformation übertragen. Ohne verlässlichen Kopf ist dieser Weg zu.
2. **Pflicht-Prüfung links-rechts.** Eine starre Transformation spiegelt nie, und ein Gehirn ist fast
   symmetrisch: ein seitenverkehrt gelesenes PET passt mit fast derselben Güte. Deshalb läuft dieselbe Suche mit
   dem links-rechts gespiegelten PET als Gegenprobe. Nur wenn das ungespiegelte PET deutlich besser passt, wird die
   Ebene gezeigt; sonst bricht der Lauf mit beiden Werten ab. Das Ergebnis steht in `bilder[].guete` neben der
   Registriergüte.

**PET, Weg 1: Verhältnis zur Referenzregion (zuerst).** Voxel geteilt durch den Median einer Referenzregion.
Braucht weder Gewicht noch Aktivität und ist zwischen Geräten robuster. Bei einem Aminosäure-Tracer ist das der
übliche Weg; die Region (etwa gesundes Hirngewebe der Gegenseite) legt der Nutzer fest, sie steht mit Serie und
Maske in der Falldatei, damit der Wert nachvollziehbar bleibt. Schwellen für Tumorvolumina übernimmt der Viewer
nicht selbst, sondern nur aus einer genannten Leitlinie, mit Quelle in der Legende.

**PET, Weg 2: SUV (körpergewichtsbezogen), nur mit Eingabe von Hand.** SUV = C · Gewicht / Aktivität zur
Aufnahmezeit. Fehlen Gewicht und Aktivität im Kopf, rechnet der Viewer SUV nur, wenn der Nutzer Gewicht,
injizierte Aktivität, Injektionszeit und Halbwertszeit selbst einträgt und `Units` Bq/ml belegt ist. Die
eingetragenen Werte stehen in der Falldatei (`bilder[].pet.suv_eingabe`) und die Legende sagt, dass sie von Hand
kommen. Liefert der Kopf alle Felder (`Units` = `BQML`, `DecayCorrection` `START` oder `ADMIN`, Gewicht,
`RadiopharmaceuticalInformationSequence` mit Dosis, Halbwertszeit und Startzeit, Aufnahmezeit aus `SeriesTime` oder
der frühesten `AcquisitionTime`), nimmt der Viewer sie.

Im Fall:

```json
{"serie": "pet_2001-04-01", "modalitaet": "PT", "datum": "2001-04-01", "ordner": "bilder/pet_2001-04-01",
 "nach_bezug": "transform/pet_2001-04-01.tfm",
 "guete": {"verfahren": "mattes_mi", "ok": true, "seitenpruefung": {"mi": 0.0, "mi_gespiegelt": 0.0, "ok": true}},
 "pet": {"tracer": "aminosaeure", "referenz": {"art": "gegenseite", "maske": "masken/pet_ref.nii.gz"},
         "suv_eingabe": null}}
```

**Umsetzung.** Die PET-Ebene läuft getrennt als OTPAS-5734 (Task 3331): `tools/mrt_overlay/pet.py` liest die
Serie, registriert sie mit Spiegelprüfung und schreibt das Verhältnis als Ebene der alten Seite. Das 8-Bit-Format
dieser Ebene (Verhältnis bis 4 in Stufen von 0,016) ist Übergang. Registrierung und Spiegelprüfung bleiben gültig;
in der App wird dieselbe Ebene als Fließkomma-NIfTI geschrieben.

## 4. Biopsie

Annahme des Entwurfs: das Material liegt als Glas-Objektträger vor, die Probe ist vor Strahlen- und
Chemotherapie entnommen.

- **Zeitleiste jetzt.** Ein Eintrag `art: "biopsie"` mit Entnahmedatum und `ausgangszustand: true`: der Verlauf
  zeigt ihn als Ausgangszustand vor der Therapie. Optional ein Verweis auf einen Befundtext im Fallordner (der
  Viewer zeigt ihn als Text, er wertet ihn nicht aus) und der Entnahmeort als Punkt oder Maske auf einer genannten
  MR-Serie (`ort: {"serie": …, "voxel_xyz": […]}`).
- **Bild erst nach Ganzschnitt-Scan.** Im Viewer erscheinen die Schnitte erst, wenn die Objektträger als
  Ganzschnitt gescannt sind (Whole Slide Imaging, `modalitaet: "SM"`). Sie bekommen eine eigene Ansicht ohne
  Registrierung auf das MRT; der Zeitleisten-Eintrag verweist dann nur auf sie. NiiVue zeigt keine Ganzschnitte;
  das braucht dann eine eigene Kachelansicht und wird erst geplant, wenn Scans da sind.

**Viewer für andere Patienten:** geparkt.

## 5. Reihenfolge der Umsetzung

Robins Entscheidung 2253: die Falldatei zuerst. Die PET-Ebene auf der alten Seite (Task 3331) läuft parallel.
Der Umstieg des Viewers (Abschnitt 6) beginnt, sobald die Falldatei steht, weil die App ihre Daten aus
`ergebnisse.viewer` findet.

| # | Aufgabe | Größe | Nutzen |
|---|---|---|---|
| 1 | Task 3320: `fall/v1` als Schema, `fall_check.py`, Demo-Fall in `workspace/offen/` | S | Alle Werkzeuge haben ein Ziel, gegen das sie testen. |
| 1' | Task 3331 (parallel): PET-Ebene auf der alten Seite, Verhältnis, Spiegelprüfung | M | PET sofort im Verlauf; Registriercode für die App. |
| 2 | Task 3321: `mrt_overlay` liest Pfade aus `fall.json` statt zehn Argumenten, schreibt `nach_bezug`, `guete`, `ergebnisse.viewer` | M | Ein Start statt zehn Pfadabfragen; Registriergüte bleibt beim Fall. |
| 3 | Task 3322: plan_mc schreibt `ergebnisse.workbench` mit Version und Hashes, RiFi-Designer schreibt `rifi` | S | Jede Dosis weiß, aus welchem Plan und Filter sie stammt. |
| 4 | Task 3323: Simulator, Brücke Einzelplan mit RiFi, `fall run` | M | Zeit mit und ohne RiFi für genau den Plan, den die Workbench rechnet. |
| 5 | App Stufe 0 und 1: Gerüst, Einzeldatei offline, Volumen als NIfTI (Abschnitt 6) | M | Beweist den Offline-Weg, bevor eine Funktion umzieht. |
| 6 | App Stufe 2 bis 5: Umzug Funktion für Funktion bis zur Gleichwertigkeit | L | Eine Bild-Engine statt Kachelbilder; Fließkomma für LETd und PET. |
| 7 | Task 3324: CT-Ebene mit HU-Fenstern, gleich in der App | S | Dosis und LETd auf der Anatomie, aus der sie gerechnet sind. |
| 8 | Task 3325: SUV mit Eingabe von Hand, in der App | S | SUV, wo Gewicht und Aktivität bekannt sind. |
| 9 | Task 3326: Biopsie als Ausgangszustand in der Zeitleiste | S | Der Verlauf zeigt, wann und wo die Probe vor der Therapie entnommen wurde. |

## 6. Umstieg des Viewers auf eine App

Robins Entscheidung: der Viewer wird von der HTML-Vorlage mit Python-Generator zu einer App. Dieser Abschnitt plant
den Umstieg; gebaut wird in den Folge-Aufgaben.

**Ziel.**

- React mit TypeScript, NiiVue als Bild-Engine (WebGL2; liest NIfTI, kann Ebenen mit Farbskala überlagern, Schnitte,
  3D-Ansicht und Messen). Eine dünne eigene React-Hülle um die NiiVue-Leinwand statt eines Wrapper-Pakets: ein
  Paket weniger, und die Hülle ist wenige Zeilen.
- Daten als echte Volumen: NIfTI auf dem Viewer-Raster, MR als int16 oder uint8, CT als int16 (HU), LETd, Dosis und
  PET-Verhältnis als float32. Keine 8-Bit-WEBP-Mosaiken mehr; Fenster und Schwellen rechnet die App auf den echten
  Werten.
- Gebaut als eine einzige HTML-Datei (Vite mit einem Einzeldatei-Plugin), damit Doppelklick offline weiter geht,
  ohne Server und ohne Netz. Versionen und Lizenzen der Pakete werden beim Bau aus der npm-Registry festgelegt.

**Wie die Daten offline in die Seite kommen.** Ein Browser lädt unter `file://` keine Nachbardatei mit `fetch`
(wahrscheinlich in Chrome und Edge, im ersten Schritt zu prüfen). Die alte Seite umgeht das mit `.js`-Dateien, die
sich per `<script>` selbst eintragen. Die App macht es ebenso: der Generator schreibt je Volumen eine `.js`-Datei,
die ein gzip-NIfTI als Base64 an die App übergibt, und eine Indexdatei aus `ergebnisse.viewer` und `zeitleiste`. Die
App gibt NiiVue das Volumen aus dem Speicher. Weg zur Reserve: der Nutzer zieht den Datenordner auf die Seite oder
wählt ihn mit einem Ordner-Dialog. Größe im Blick: ein float32-Volumen von 256³ Voxeln ist 64 MB roh; die erste Stufe
misst, wie groß ein echter Verlauf gepackt wird, und ob LETd und PET auf einem gröberen Raster reichen.

**Eigenständige App oder Teil der Workbench-Web-UI.**

| | eigenständige App (Einzeldatei) | Teil der Workbench-Web-UI |
|---|---|---|
| Offline per Doppelklick | ja | nein, braucht den laufenden Server |
| Patientendaten | bleiben im Ordner des Nutzers, kein Server liest sie | gehen durch den Workbench-Server, der `lokal` und `robin` bewusst nicht ausliefert |
| Bau | Node-Bau nur für die App; der Generator bleibt Python | die Web-UI ist heute eine Seite `ui.html` ohne Bau-Schritt auf der Standardbibliothek; React brächte einen Bau in die ganze Workbench |
| Ein Ort für Plan, Dosis und Bild | nein, zwei Fenster | ja |

Entschieden im Entwurf: **eigenständige App**. Sie hält den Doppelklick offline und die Datenzonen sauber, und die
Workbench bleibt ohne Bau-Schritt. Die Workbench-Web-UI bekommt später nur einen Knopf, der den Viewer eines Falls
öffnet; für den Demo-Fall in `workspace/offen/` kann der Server die gebaute Datei mit ausliefern. Leicht umzukehren,
solange nur das Gerüst steht: die App-Komponenten hängen nicht davon ab, wer sie lädt.

**Wo die App liegt.** Quellen in `tools/mrt_viewer/` (eigenes `package.json`), die gebaute Einzeldatei wird
mit eingecheckt, damit der Python-Generator auf einem Rechner ohne Node arbeitet. Im Repo nur Code und der
Demo-Fall aus `workspace/offen/`; keine Daten, Pfade oder Werte aus Aufnahmen. Ob die CI den Bau prüft, entscheidet
die Aufgabe der Stufe 0 nach der CI-Regel des Projekts.

**Umzug Funktion für Funktion.** Die alte `Verlauf.html` bleibt, bis die App jede ihrer Funktionen kann. Liste der
Gleichwertigkeit:

| Stufe | Funktion | auf der alten Seite |
|---|---|---|
| 0 | Gerüst, Einzeldatei, ein NIfTI offline per Doppelklick geladen | |
| 1 | Generator schreibt NIfTI-Volumen und Index aus `fall.json` | Kachelbilder bleiben parallel |
| 2 | Zeitachse, Schnittansicht, Serienwahl | Zeitachse |
| 3 | Ebenen: LETd, Dosis, Ventrikelsaum, PET-Verhältnis, mit Farbskala und Legende | LETd-Ebene, PET-Ebene |
| 4 | Lineal, 3D-Ansicht | Lineal, 3D |
| 5 | Vergleich zweier Zeitpunkte, Blinken, Differenz | Vergleich, Blinken, Differenz |
| 6 | Laborwerte, Befundtexte, Ereignisse, Biopsie | Laborwerte |
| 7 | neu: CT mit HU-Fenstern, SUV mit Eingabe von Hand | fehlt dort |

Gleichwertig ist die App, wenn jede Zeile der Spalte "auf der alten Seite" auf dem Demo-Fall und auf einem echten
Verlauf dasselbe zeigt. Dann entscheidet Robin, ob die alte Seite geht.

**Was an der alten Seite nur Übergang ist.** Die PET-Ebene aus Task 3331 (8-Bit-Verhältnis) und die LETd-Ebene mit
`letd_all.js` bleiben bis Stufe 3 in Gebrauch; an ihrem Ausgabeformat wird nichts mehr ausgebaut. Ihr
Rechen-Code (`pet.py`, `layer.py`, `register.py`) bleibt und schreibt zusätzlich NIfTI. Neue Ebenen (CT, SUV) kommen
nur in die App.
