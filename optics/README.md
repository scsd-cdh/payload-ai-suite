# Zetane Optical Payload — RayOptics Modeling (Issue #58)

This folder models the Zetane CubeSat camera optics: a **Schneider TOPAZ 2.0/30 C** lens on a **2048 × 2048, 5.5 µm** sensor, imaging Earth from **400 km (nominal)** and **450 km (worst case)**.

> **Important:** The real TOPAZ optical design (glass types, number of lenses, surface shapes) is not public. `rayoptics_model.py` uses a **two-lens stand-in** that matches the datasheet's focal length and sensor distance. Its image-quality results (spot size, MTF, aberrations) describe the **stand-in only**, not the real lens. Use Schneider's MTF charts or lab measurements to judge the real lens.

---

## Files

| File | What it does |
|---|---|
| `optical_model.py` | Quick first-order calculations (FOV, GSD, swath) and an ideal thin-lens sketch |
| `rayoptics_model.py` | Builds the stand-in lens in RayOptics and runs all analyses (Steps 3–6) |
| `fit_equivalent_lens.py` | Re-derives the stand-in lens shape from the datasheet (optional, about 1 minute) |
| `results/` | Figures and CSV files written by `rayoptics_model.py` |

## Setup

```bash
git checkout optics-model-simulation        # or main after PR #57 is merged
git checkout -b feature/rayoptics-analysis
python -m venv venv
source venv/bin/activate                    # Windows: venv\Scripts\activate
pip install rayoptics numpy scipy matplotlib
```

Tested with Python 3.13 and RayOptics 0.9.10.

**First run:** RayOptics's glass library downloads the refractiveindex.info database (from GitHub) the first time it is imported, so you need an internet connection once.

## Run

```bash
python optics/optical_model.py          # quick numbers + simple diagram
python optics/rayoptics_model.py        # full model -> optics/results/
python optics/fit_equivalent_lens.py    # optional: refit the stand-in lens
```

---

## Inputs and where they come from

| Value | Source |
|---|---|
| Focal length 29.99 mm (nominal 30) | TOPAZ datasheet (f'eff) |
| f/2 (widest aperture; range f/2–f/16) | TOPAZ datasheet |
| Entrance pupil diameter 15.0 mm | = focal length ÷ f-number |
| SF −4.91, S'F' 13.96, SEP 17.05, S'AP −27.00, β'P 1.37, HH' −11.95, Σd 29.17 mm | TOPAZ datasheet (first-order data) |
| Image circle 19.3 mm, max angle of view 37° | TOPAZ datasheet |
| Wavelengths 425–675 nm, weights 8/16/23/22/19/13 % | TOPAZ datasheet "VIS" spectrum (same as their MTF charts) |
| 2048 × 2048 px, 5.5 µm pitch, 11.264 × 11.264 mm, 15.93 mm diagonal | Camera documentation |
| 400 km nominal, 450 km worst-case altitude | Team decision (PR #57 review) |

## Assumptions (not from any document)

1. **Stand-in lens:** two N-BK7 singlets with the aperture stop between them. The real lens has more elements and different glasses.
2. **Fitting priorities:** focal length and back focal distance are matched exactly, and total length and f/2 are enforced. Front focal distance, pupil positions and pupil magnification are best-fit, and the fit also keeps spot size small and every lens edge at least 1 mm thick. See `fit_equivalent_lens.py` for the weights.
3. **Object at infinity:** Earth is treated as infinitely far away.
4. **Nadir pointing and flat Earth:** the camera points straight down, and Earth curvature is ignored. At 400 km and a ±10.6° half field, this approximation is small.
5. **Sensor at best focus:** the sensor sits where the on-axis polychromatic spot is smallest (720 µm in front of the paraxial focus for this stand-in).
6. **Field points:** on-axis (0 mm), intermediate (2.82 mm), sensor edge (5.63 mm) and sensor corner (7.97 mm). Angles use θ = atan(h / f).

---

## Results

### How well the stand-in matches the datasheet

| Quantity | Datasheet | Stand-in | Status |
|---|---|---|---|
| EFL (mm) | 29.99 | 29.99 | matched |
| S'F' back focal distance (mm) | 13.96 | 13.96 | matched |
| f-number | 2.0 | 2.0 | matched |
| Σd lens length (mm) | 29.17 | 29.17 | matched |
| SF front focal distance (mm) | −4.91 | −8.08 | best fit |
| SEP entrance pupil (mm) | 17.05 | 16.02 | best fit |
| S'AP exit pupil (mm) | −27.00 | −23.36 | best fit |
| Pupil magnification | 1.37 | 1.24 | best fit |
| HH' principal planes (mm) | −11.95 | −8.77 | check only |

### Geometry checks
- The sensor diagonal (15.93 mm) fits inside the 19.3 mm image circle, and the corner field (±14.9°) is inside the lens's 37° maximum angle of view.
- No ray-trace failures occurred at any field. The model has no mechanical apertures other than the stop, so it **cannot predict vignetting**. The datasheet shows relative illumination falling to about 50% at the 9.7 mm image-circle edge at f/2 (read from chart).

### Image quality of the stand-in (best focus, TOPAZ VIS spectrum)

| Field | Angle | RMS spot radius | RMS spot diameter (pixels) |
|---|---|---|---|
| On-axis | 0.0° | 60 µm | 22 px |
| Intermediate | 5.4° | 65 µm | 24 px |
| Edge | 10.6° | 103 µm | 38 px |
| Corner | 14.9° | 166 µm | 60 px |

- **Spot size vs pixel:** every stand-in spot is much larger than one 5.5 µm pixel. The main causes are spherical aberration and axial color: two N-BK7 lenses cannot correct color, and blue (425 nm) and red (675 nm) focus about 0.6 mm apart.
- **Field curvature:** best focus moves toward the lens as image height increases (`04_field_curvature_distortion.png`), so the edges and corners are blurrier than the center.
- **Distortion:** barrel distortion of −1.2% at the sensor corner and −1.9% at the 9.65 mm image-circle edge. Schneider's chart shows roughly −2.5% to −3% at 9.7 mm (read by eye), so the stand-in has the same sign and similar size.
- **Lateral color:** 6.6 µm (about 1.2 px) between 425 and 675 nm at the corner.
- **MTF:** the geometric × diffraction MTF estimate is close to zero (≤ 0.13) at 20–91 lp/mm for every field. Schneider's charts show about 90% at 20 lp/mm, 73% at 40 lp/mm and 50% at 80 lp/mm near the center at f/2 (read from chart, at β' = −0.01). **Conclusion: this stand-in cannot support a defensible MTF prediction for the real lens.** Use the manufacturer charts instead, or measure the real lens. The sensor Nyquist frequency is 90.9 lp/mm.

### Mission numbers (nadir, flat Earth)

| | 400 km (nominal) | 450 km (worst case) |
|---|---|---|
| GSD | 73.4 m/pixel | 82.5 m/pixel |
| Swath | 150.2 × 150.2 km | 169.0 × 169.0 km |
| FOV | 21.3° × 21.3° (29.8° diagonal) | same |
| Stand-in corner distortion shift | 17 px ≈ 1.3 km | 17 px ≈ 1.4 km |

`optical_model.py` uses the nominal 30 mm focal length, so its GSD is about 0.03 m smaller than these values.

### What this means for wildfire detection

- **Resolution:** at 73–83 m GSD, a fire much smaller than one pixel can only be seen if it is hot and bright enough to raise that pixel's signal. Optical blur spreads that signal into neighbouring pixels and lowers the peak, so a sharper lens detects smaller or cooler fires.
- **Blur:** Schneider's MTF at Nyquist (~45–50% near the center at 80 lp/mm, read from chart) suggests the real lens is roughly matched to the pixel size near the center, with more blur toward the edges and corners (tangential MTF drops sharply near 9.7 mm). The stand-in's blur is much larger and should **not** be used for detection estimates.
- **Distortion:** a 1–3% barrel distortion moves features near the corners by tens of pixels (more than 1 km). Without correction, fire locations near the image edges would be placed in the wrong spot, so distortion should be calibrated and corrected in geolocation.
- **Pointing uncertainty:** the ground offset at nadir is about altitude × tan(error). For example, 0.1° gives about 0.70 km (≈ 10 px) at 400 km and 0.79 km at 450 km. The real pointing error is not yet known (see open questions).
- **Edge vs center:** brightness falls off and blur grows toward the corners, so detection is likely less sensitive at the image edges than at the center.

---

## Limitations

- The stand-in lens is **not** the TOPAZ design. Its aberrations, spot sizes and MTF are not representative of the real lens.
- Schneider's MTF and distortion charts are for finite conjugates (β' = −0.01 to −0.05, object about 0.6–6 m away), not infinity. Values quoted above were read by eye from the charts.
- Only visible wavelengths (425–675 nm) are modelled. The lens coating also covers 700–1000 nm, but there are no NIR MTF data.
- No vignetting, stray light, thermal or vacuum focus-shift, detector MTF, motion blur or atmospheric effects are modelled.
- The geometric MTF uses a finite ray grid, so values below ~0.05 are at the noise level.

## Open questions

1. Can the team get the real TOPAZ prescription or a Zemax "black box" file from Schneider? RayOptics can import `.zmx` files, which would replace the stand-in.
2. Which spectral band will the mission use: visible, NIR, or a specific filter?
3. What is the required GSD or smallest fire size from the CDR?
4. What is the expected pointing knowledge and accuracy of the ADCS?
5. The CDR value for altitude was quoted as 417 km in the PR #57 review. Confirm that 400 km nominal and 450 km worst case are current.

## Recommendations for physical testing

1. **Focus at infinity:** use a collimator to set and lock focus, then re-check after thermal cycling and in vacuum, because focus can shift in space.
2. **MTF and sharpness:** measure with a slanted-edge target (ISO 12233) at center, edge and corner, at f/2 and in the chosen spectral band.
3. **Distortion:** image a dot or checkerboard grid and fit a distortion model for geolocation.
4. **Relative illumination:** image a uniform light source (integrating sphere or flat panel) to build a flat-field correction.
5. **Spectral response:** measure through the flight filter, if one is used.
6. **Focal length:** check the real focal length from the grid images, because GSD scales directly with it.
