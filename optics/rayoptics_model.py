"""
Zetane optical payload - RayOptics equivalent-lens model (Issue #58)

WHAT THIS IS
    A simplified two-element stand-in for the Schneider TOPAZ 2.0/30 C lens.
    It is solved so that its first-order (paraxial) data match the TOPAZ
    datasheet: focal length, back focal distance, front focal distance,
    entrance-pupil position and total lens length.

WHAT THIS IS NOT
    It is NOT Schneider's real optical design (the real prescription is not
    public). Aberrations, spot sizes and MTF computed here describe the
    stand-in only. Use the manufacturer MTF charts or lab measurements to
    judge the real lens.

RUN
    python optics/rayoptics_model.py
    -> figures and CSV files are written to optics/results/
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # write figures to files
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from rayoptics.environment import (
    OpticalModel, PupilSpec, FieldSpec, WvlSpec, InteractiveLayout,
)
from rayoptics.raytr.trace import trace_base

# =============================================================================
# 1. INPUTS
# =============================================================================
# --- Lens: Schneider TOPAZ 2.0/30 C datasheet (ID 1078946, 7/2025) ----------
EFL_MM = 29.99            # f'eff, effective focal length
F_NUMBER = 2.0            # widest aperture (F/2 ... F/16)
SF_MM = -4.91             # first lens vertex -> object-space focal point
SPF_MM = 13.96            # S'F': last lens vertex -> image focal point (BFL)
SEP_MM = 17.05            # first lens vertex -> entrance pupil
SPAP_MM = -27.00          # S'AP: last lens vertex -> exit pupil (check only)
PUPIL_MAG = 1.37          # beta'P = exit pupil / entrance pupil (check only)
SUM_D_MM = 29.17          # first -> last lens vertex (total glass length)
HHP_MM = -11.95           # HH': principal plane separation (check only)
IMAGE_CIRCLE_MM = 19.3    # max sensor size (image circle diameter)
EPD_MM = EFL_MM / F_NUMBER  # entrance pupil diameter (~15 mm)

# Datasheet "VIS" spectrum used for Schneider's MTF charts
WAVELENGTHS_NM = [425.0, 475.0, 525.0, 575.0, 625.0, 675.0]
WAVELENGTH_WEIGHTS = [8, 16, 23, 22, 19, 13]          # percent
REF_WVL_INDEX = 2                                       # 525 nm (largest weight)

# --- Camera: from camera documentation (confirmed in PR #57 review) ---------
SENSOR_PIXELS = (2048, 2048)
PIXEL_PITCH_MM = 0.0055                                 # 5.5 um

# --- Mission -----------------------------------------------------------------
ORBIT_ALTITUDES_KM = {"nominal": 400.0, "worst case": 450.0}

# --- ASSUMPTIONS (not from any document) -------------------------------------
# The real TOPAZ glass types, number of elements and surface shapes are
# unknown. The stand-in uses two N-BK7 singlets (shapes and thicknesses are
# solved to match the datasheet first-order data, see section 2).
GLASS = ("N-BK7", "Schott")

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Derived sensor geometry
sensor_w = SENSOR_PIXELS[0] * PIXEL_PITCH_MM            # 11.264 mm
sensor_h = SENSOR_PIXELS[1] * PIXEL_PITCH_MM            # 11.264 mm
half_diag = 0.5 * np.hypot(sensor_w, sensor_h)          # 7.965 mm
nyquist_lpmm = 1.0 / (2 * PIXEL_PITCH_MM)               # 90.9 lp/mm

# Fields: image heights on the sensor -> object angles
FIELD_NAMES = ["On-axis", "Intermediate", "Edge", "Corner"]
FIELD_IMG_HT = [0.0, 0.25 * sensor_w, 0.5 * sensor_w, half_diag]
FIELD_ANGLES = [float(np.degrees(np.arctan(h / EFL_MM))) for h in FIELD_IMG_HT]


# =============================================================================
# 2. BUILD THE STAND-IN LENS
#    Two N-BK7 singlets with the aperture stop between them:
#      [lens 1 front, lens 1 back, STOP, lens 2 front, lens 2 back] -> sensor
#    The shape below comes from fit_equivalent_lens.py, which matches EFL and
#    S'F' exactly, best-fits SF / SEP / S'AP / pupil magnification, keeps the
#    spot size reasonable and keeps every lens edge >= 1 mm thick.
#    Total length = Sum d (29.17 mm) and f/2 are enforced by construction.
# =============================================================================
LENS_PRESCRIPTION = {
    "c1": 0.04787318, "c2": 0.01267523,     # lens 1 curvatures (1/mm)
    "c3": 0.03188793, "c4": -0.02449652,    # lens 2 curvatures (1/mm)
    "gap_lens1_stop": 9.573788,
    "t1": 3.721332, "t2": 3.355814,
}


def build_model(lp=LENS_PRESCRIPTION, flds=None, wvls=None):
    g2 = SUM_D_MM - lp["t1"] - lp["t2"] - lp["gap_lens1_stop"]
    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    osp["pupil"] = PupilSpec(osp, key=["object", "epd"], value=EPD_MM)
    osp["fov"] = FieldSpec(osp, key=["object", "angle"],
                           flds=FIELD_ANGLES if flds is None else flds)
    osp["wvls"] = WvlSpec(list(zip(WAVELENGTHS_NM, WAVELENGTH_WEIGHTS)),
                          ref_wl=REF_WVL_INDEX)
    opm.radius_mode = False                       # surfaces given as curvature
    sm.gaps[0].thi = 1e10                         # object (Earth) at infinity
    sm.add_surface([lp["c1"], lp["t1"], *GLASS])
    sm.add_surface([lp["c2"], lp["gap_lens1_stop"]])
    sm.add_surface([0.0, g2])                     # aperture stop
    sm.set_stop()
    sm.add_surface([lp["c3"], lp["t2"], *GLASS])
    sm.add_surface([lp["c4"], 0.0])               # image gap set below
    opm.update_model()
    # put the sensor at the paraxial focus
    sm.gaps[-1].thi = opm["analysis_results"]["parax_data"].fod.bfl
    opm.update_model()
    return opm


def first_order(opm):
    return opm["analysis_results"]["parax_data"].fod


opm = build_model()
sm, osp = opm["seq_model"], opm["optical_spec"]
fod = first_order(opm)
WVLS = osp["wvls"]
REF_WVL = WVLS.central_wvl

# Compare with datasheet
_pr = opm["parax_model"].pr
_k = len(sm.ifcs) - 2                            # last lens surface
exp_dist = -_pr[_k][0] / _pr[_k][1]              # S'AP from chief ray
pupil_mag = fod.exp_radius / fod.enp_radius
h_abs = fod.pp1                                   # H from first vertex
hp_abs = SUM_D_MM + fod.ppk                       # H' from first vertex
hhp = hp_abs - h_abs

match_rows = [
    ("EFL f'eff (mm)", EFL_MM, fod.efl, "fitted (exact)"),
    ("S'F' back focal distance (mm)", SPF_MM, fod.bfl, "fitted (exact)"),
    ("f-number", F_NUMBER, fod.fno, "set by EPD"),
    ("Sum d lens length (mm)", SUM_D_MM, sum(g.thi for g in sm.gaps[1:-1]),
     "set by construction"),
    ("SF front focal distance (mm)", SF_MM, fod.ffl, "best fit"),
    ("SEP entrance pupil (mm)", SEP_MM, fod.enp_dist, "best fit"),
    ("S'AP exit pupil (mm)", SPAP_MM, exp_dist, "best fit"),
    ("Pupil magnification", PUPIL_MAG, pupil_mag, "best fit"),
    ("HH' principal planes (mm)", HHP_MM, hhp, "check only"),
]

print("=" * 70)
print("ZETANE - TOPAZ 2.0/30 C EQUIVALENT LENS (RayOptics)")
print("=" * 70)
sm.list_model()
LP = LENS_PRESCRIPTION
c1, c2, c3, c4 = LP["c1"], LP["c2"], LP["c3"], LP["c4"]
g1, t1, t2 = LP["gap_lens1_stop"], LP["t1"], LP["t2"]
g2 = SUM_D_MM - t1 - t2 - g1
R = [1 / c if abs(c) > 1e-12 else np.inf for c in (c1, c2, c3, c4)]
print(f"\nSolved stand-in: lens 1 R={R[0]:.3f}/{R[1]:.3f} mm t={t1:.3f} mm, "
      f"lens 2 R={R[2]:.3f}/{R[3]:.3f} mm t={t2:.3f} mm, gaps {g1:.3f}/{g2:.3f} mm")
print(f"\n{'Quantity':34s}{'Datasheet':>11s}{'Model':>11s}   Status")
for name, ds, mod, status in match_rows:
    print(f"{name:34s}{ds:11.3f}{mod:11.3f}   {status}")
print(f"\nSensor {sensor_w:.3f} x {sensor_h:.3f} mm, diagonal {2*half_diag:.2f} mm "
      f"(image circle {IMAGE_CIRCLE_MM} mm -> "
      f"{'fits' if 2*half_diag <= IMAGE_CIRCLE_MM else 'DOES NOT FIT'})")
print("Fields: " + ", ".join(f"{n} {h:.2f} mm / {a:.2f} deg"
                              for n, h, a in zip(FIELD_NAMES, FIELD_IMG_HT, FIELD_ANGLES)))


# =============================================================================
# 3. RAY TRACING HELPERS
# =============================================================================
def trace_xy(opm_, fld, wvl, pupil):
    """Image-plane (x, y) of one ray, or None if the ray fails."""
    try:
        return trace_base(opm_, pupil, fld, wvl)[0][-1][0][:2]
    except Exception:
        return None


def spot(opm_, fi, num=21):
    """Polychromatic spot: arrays x, y (mm) and weights, plus failure count."""
    fld = opm_["optical_spec"]["fov"].fields[fi]
    xs, ys, ws, wl_idx, failed = [], [], [], [], 0
    grid = np.linspace(-1, 1, num)
    for k, (wvl, wt) in enumerate(zip(WAVELENGTHS_NM, WAVELENGTH_WEIGHTS)):
        for px in grid:
            for py in grid:
                if px**2 + py**2 > 1:
                    continue
                p = trace_xy(opm_, fld, wvl, [px, py])
                if p is None:
                    failed += 1
                    continue
                xs.append(p[0]); ys.append(p[1]); ws.append(wt); wl_idx.append(k)
    return np.array(xs), np.array(ys), np.array(ws, float), np.array(wl_idx), failed


def rms_radius(x, y, w):
    cx, cy = np.average(x, weights=w), np.average(y, weights=w)
    return np.sqrt(np.average((x - cx)**2 + (y - cy)**2, weights=w))


# =============================================================================
# 4. FOCUS: paraxial focus vs best focus (smallest on-axis polychromatic spot)
# =============================================================================
def rms_at_shift(dz):
    sm.gaps[-1].thi = fod.bfl + dz
    opm.update_model()
    x, y, w, _, _ = spot(opm, 0, num=15)
    return rms_radius(x, y, w)

rms_paraxial = rms_at_shift(0.0)
best = minimize_scalar(rms_at_shift, bounds=(-3.0, 1.0), method="bounded")
BEST_SHIFT = best.x
sm.gaps[-1].thi = fod.bfl + BEST_SHIFT
opm.update_model()
print(f"\nBest focus: sensor moved {BEST_SHIFT*1000:+.0f} um from paraxial focus "
      f"(on-axis RMS {rms_paraxial*1000:.1f} -> {best.fun*1000:.1f} um)")


# =============================================================================
# 5. RAY LAYOUT + RAY-FAILURE CHECK
# =============================================================================
fig = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
                 do_draw_rays=True, do_paraxial_layout=False,
                 is_dark=False, figsize=(11, 5)).plot()
fig.suptitle("TOPAZ equivalent lens - ray layout (0, intermediate, edge, corner fields)")
fig.savefig(OUT_DIR / "01_layout.png", dpi=150)
plt.close(fig)


# =============================================================================
# 6. SPOT DIAGRAMS + RMS SPOT SIZE vs PIXEL
# =============================================================================
wl_colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(WAVELENGTHS_NM)))
spot_rows = []
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
print("\nSpot sizes at best focus (polychromatic, weighted) vs 5.5 um pixel:")
for fi, ax in enumerate(axes):
    x, y, w, k, failed = spot(opm, fi)
    cx, cy = np.average(x, weights=w), np.average(y, weights=w)
    dx, dy = (x - cx) * 1000, (y - cy) * 1000
    rms = rms_radius(x, y, w) * 1000
    geo = np.max(np.hypot(dx, dy))
    for j, wl in enumerate(WAVELENGTHS_NM):
        m = k == j
        ax.scatter(dx[m], dy[m], s=2, color=wl_colors[j], label=f"{wl:.0f} nm")
    p = PIXEL_PITCH_MM * 1000
    ax.add_patch(plt.Rectangle((-p/2, -p/2), p, p, fill=False, color="black",
                               lw=1.2, label="1 pixel (5.5 um)"))
    ax.set_title(f"{FIELD_NAMES[fi]} ({FIELD_ANGLES[fi]:.1f} deg)\n"
                 f"RMS radius {rms:.1f} um = {2*rms/p:.1f} px dia.")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    spot_rows.append({
        "field": FIELD_NAMES[fi], "image_height_mm": round(FIELD_IMG_HT[fi], 3),
        "angle_deg": round(FIELD_ANGLES[fi], 3),
        "rms_radius_um": round(rms, 2), "geo_radius_um": round(geo, 2),
        "rms_diameter_px": round(2 * rms / p, 2), "failed_rays": failed,
    })
    print(f"  {FIELD_NAMES[fi]:12s} {FIELD_ANGLES[fi]:6.2f} deg  RMS radius "
          f"{rms:7.2f} um  = {2*rms/p:6.2f} px diameter   failed rays: {failed}")
axes[0].legend(fontsize=6, loc="upper left", markerscale=3)
fig.suptitle("Spot diagrams at best focus (TOPAZ VIS wavelengths & weights)")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_spot_diagrams.png", dpi=150)
plt.close(fig)


# =============================================================================
# 7. RAY ABERRATION FANS (tangential, relative to chief ray at 525 nm)
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
pupil = np.linspace(-1, 1, 41)
for fi, ax in enumerate(axes):
    fld = osp["fov"].fields[fi]
    chief_ref = trace_xy(opm, fld, REF_WVL, [0, 0])
    for j, wl in enumerate(WAVELENGTHS_NM):
        ey = []
        for py in pupil:
            pnt = trace_xy(opm, fld, wl, [0, py])
            ey.append(np.nan if pnt is None else (pnt[1] - chief_ref[1]) * 1000)
        ax.plot(pupil, ey, color=wl_colors[j], label=f"{wl:.0f} nm")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(FIELD_NAMES[fi]); ax.set_xlabel("Relative pupil height")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Transverse ray error (um)")
axes[0].legend(fontsize=6)
fig.suptitle("Tangential ray aberration fans at best focus")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_ray_fans.png", dpi=150)
plt.close(fig)


# =============================================================================
# 8. FIELD CURVATURE, DISTORTION, CHROMATIC FOCUS SHIFT (at paraxial plane)
# =============================================================================
def focus_crossing(r_a, r_b, axis):
    """z (from current image plane) where two rays cross in the given axis."""
    (pa, da), (pb, db) = r_a, r_b
    ma, mb = da[axis] / da[2], db[axis] / db[2]
    return -(pa[axis] - pb[axis]) / (ma - mb)


def ray_end(opm_, fld, wvl, pupil_):
    r = trace_base(opm_, pupil_, fld, wvl)[0][-1]
    return r[0], r[1]


img_heights = np.linspace(0, IMAGE_CIRCLE_MM / 2, 16)   # out to 9.65 mm
angles = np.degrees(np.arctan(img_heights / EFL_MM))
curv_t, curv_s, dist_pct = [], [], []
opm_scan = build_model(flds=list(angles))      # sensor at paraxial focus
osp_scan = opm_scan["optical_spec"]
for i, a in enumerate(angles):
    fld = osp_scan["fov"].fields[i]
    chief_p, _ = ray_end(opm_scan, fld, REF_WVL, [0, 0])
    y_ideal = EFL_MM * np.tan(np.radians(a))
    dist_pct.append(0.0 if a == 0 else 100 * (abs(chief_p[1]) - y_ideal) / y_ideal)
    curv_t.append(focus_crossing(ray_end(opm_scan, fld, REF_WVL, [0, 0.05]),
                                 ray_end(opm_scan, fld, REF_WVL, [0, -0.05]), 1) * 1000)
    curv_s.append(focus_crossing(ray_end(opm_scan, fld, REF_WVL, [0.05, 0]),
                                 ray_end(opm_scan, fld, REF_WVL, [-0.05, 0]), 0) * 1000)

chroma = {}
fld0 = osp_scan["fov"].fields[0]
for wl in WAVELENGTHS_NM:
    chroma[wl] = focus_crossing(ray_end(opm_scan, fld0, wl, [0, 0.05]),
                                ray_end(opm_scan, fld0, wl, [0, -0.05]), 1) * 1000
lateral_color_um = []
fld_corner = None
for i, h in enumerate(img_heights):
    if h >= half_diag:
        fld_corner = osp_scan["fov"].fields[i]
        break
if fld_corner is None:
    fld_corner = osp_scan["fov"].fields[-1]
y_blue = ray_end(opm_scan, fld_corner, WAVELENGTHS_NM[0], [0, 0])[0][1]
y_red = ray_end(opm_scan, fld_corner, WAVELENGTHS_NM[-1], [0, 0])[0][1]
lat_color_corner_um = (y_red - y_blue) * 1000

dist_at_corner = float(np.interp(half_diag, img_heights, dist_pct))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.plot(curv_t, img_heights, "o-", label="Tangential")
ax1.plot(curv_s, img_heights, "s--", label="Sagittal")
ax1.axvline(BEST_SHIFT * 1000, color="green", ls=":", label="Sensor (best focus)")
ax1.axhline(half_diag, color="gray", ls=":", lw=0.8)
ax1.text(ax1.get_xlim()[0], half_diag, " sensor corner", va="bottom", fontsize=8)
ax1.set_xlabel("Focus position relative to paraxial image plane (um)")
ax1.set_ylabel("Image height (mm)"); ax1.set_title("Field curvature (525 nm)")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax2.plot(dist_pct, img_heights, "o-", color="tab:purple", label="Stand-in model")
ax2.axhline(half_diag, color="gray", ls=":", lw=0.8)
ax2.axvline(0, color="gray", lw=0.5)
ax2.set_xlim(-5, 5)
ax2.set_xlabel("Distortion (%)"); ax2.set_title("Distortion (compare with datasheet p.3)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_field_curvature_distortion.png", dpi=150)
plt.close(fig)

print(f"\nDistortion at sensor corner ({half_diag:.2f} mm): {dist_at_corner:+.2f} %")
print(f"Distortion at image-circle edge ({img_heights[-1]:.2f} mm): {dist_pct[-1]:+.2f} %")
print("Axial colour (focus shift vs paraxial plane): " + ", ".join(
    f"{wl:.0f} nm {s:+.0f} um" for wl, s in chroma.items()))
print(f"Lateral colour at corner (675 - 425 nm): {lat_color_corner_um:+.1f} um")


# =============================================================================
# 9. MTF ESTIMATE (geometric MTF x diffraction MTF) - stand-in only
#    Geometric MTF = |weighted average of exp(-2*pi*i*nu*position)| over rays.
#    Valid when blur >> diffraction spot, which is the case here.
# =============================================================================
def diffraction_mtf(nu, wvl_nm):
    cutoff = 1.0 / (wvl_nm * 1e-6 * F_NUMBER)          # lp/mm
    v = np.clip(nu / cutoff, 0, 1)
    return (2 / np.pi) * (np.arccos(v) - v * np.sqrt(1 - v**2))


def mtf(x, y, w, kidx, nu, axis):
    pos = (y if axis == "tan" else x)
    pos = pos - np.average(pos, weights=w)
    geo = np.abs(np.sum(w * np.exp(-2j * np.pi * nu * pos)) / np.sum(w))
    diff = np.sum([WAVELENGTH_WEIGHTS[j] * diffraction_mtf(nu, WAVELENGTHS_NM[j])
                   for j in range(len(WAVELENGTHS_NM))]) / np.sum(WAVELENGTH_WEIGHTS)
    return geo * diff

FREQS = [20.0, 40.0, 80.0, round(nyquist_lpmm, 1)]
mtf_rows = []
print("\nMTF estimate at best focus (stand-in lens):")
for fi in range(4):
    x, y, w, k, _ = spot(opm, fi, num=41)
    row = {"field": FIELD_NAMES[fi], "image_height_mm": round(FIELD_IMG_HT[fi], 2)}
    for nu in FREQS:
        row[f"tan_{nu:g}"] = round(mtf(x, y, w, k, nu, "tan"), 3)
        row[f"sag_{nu:g}"] = round(mtf(x, y, w, k, nu, "sag"), 3)
    mtf_rows.append(row)
    print(f"  {FIELD_NAMES[fi]:12s} " + "  ".join(
        f"{nu:g} lp/mm T{row[f'tan_{nu:g}']:.2f}/S{row[f'sag_{nu:g}']:.2f}" for nu in FREQS))


# =============================================================================
# 10. MISSION GEOMETRY (nadir pointing, flat-Earth approximation)
# =============================================================================
fov_x = 2 * np.degrees(np.arctan(0.5 * sensor_w / EFL_MM))
fov_y = 2 * np.degrees(np.arctan(0.5 * sensor_h / EFL_MM))
fov_d = 2 * np.degrees(np.arctan(half_diag / EFL_MM))
corner_shift_px = abs(dist_at_corner / 100 * half_diag) / PIXEL_PITCH_MM
mission_rows = []
print("\nMission geometry (nadir, flat Earth):")
for label, h_km in ORBIT_ALTITUDES_KM.items():
    scale = h_km * 1000 / EFL_MM                 # metres on ground per mm on sensor
    gsd = PIXEL_PITCH_MM * scale
    row = {"case": label, "altitude_km": h_km, "gsd_m": round(gsd, 2),
           "swath_x_km": round(sensor_w * scale / 1000, 2),
           "swath_y_km": round(sensor_h * scale / 1000, 2),
           "fov_x_deg": round(fov_x, 2), "fov_y_deg": round(fov_y, 2),
           "fov_diag_deg": round(fov_d, 2),
           "corner_distortion_shift_px": round(corner_shift_px, 1),
           "corner_distortion_shift_km": round(corner_shift_px * gsd / 1000, 2)}
    for r in spot_rows:
        row[f"blur_rms_dia_{r['field'].lower()}_m"] = round(
            2 * r["rms_radius_um"] / 1000 * scale, 1)
    mission_rows.append(row)
    print(f"  {label:10s} {h_km:.0f} km: GSD {gsd:.2f} m/px, swath "
          f"{row['swath_x_km']:.1f} x {row['swath_y_km']:.1f} km, FOV {fov_x:.2f} deg "
          f"(diag {fov_d:.2f})")
    print("             optical blur (RMS dia.) on ground: " + ", ".join(
        f"{r['field']} {row['blur_rms_dia_' + r['field'].lower() + '_m']:.0f} m"
        for r in spot_rows))
    print(f"             corner distortion shift: {corner_shift_px:.1f} px "
          f"= {row['corner_distortion_shift_km']:.2f} km on ground")


# =============================================================================
# 11. SAVE NUMBERS
# =============================================================================
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)

write_csv(OUT_DIR / "datasheet_match.csv",
          [{"quantity": n, "datasheet": ds, "model": round(m, 4), "status": s}
           for n, ds, m, s in match_rows])
write_csv(OUT_DIR / "spot_sizes.csv", spot_rows)
write_csv(OUT_DIR / "mtf_estimate.csv", mtf_rows)
write_csv(OUT_DIR / "mission_geometry.csv", mission_rows)
write_csv(OUT_DIR / "distortion_curve.csv",
          [{"image_height_mm": round(h, 3), "field_angle_deg": round(a, 3),
            "distortion_pct": round(d, 3), "tangential_focus_um": round(ct, 1),
            "sagittal_focus_um": round(cs, 1)}
           for h, a, d, ct, cs in zip(img_heights, angles, dist_pct, curv_t, curv_s)])
with open(OUT_DIR / "lens_prescription.txt", "w") as f:
    f.write("Stand-in lens (NOT the real TOPAZ prescription)\n")
    f.write(f"Lens 1: N-BK7 R1={R[0]:.4f} R2={R[1]:.4f} t={t1:.4f} mm\n")
    f.write(f"Gap lens 1 -> stop: {g1:.4f} mm\nGap stop -> lens 2: {g2:.4f} mm\n")
    f.write(f"Lens 2: N-BK7 R1={R[2]:.4f} R2={R[3]:.4f} t={t2:.4f} mm\n")
    f.write(f"Lens 2 -> sensor (paraxial): {fod.bfl:.4f} mm, "
            f"best focus shift {BEST_SHIFT*1000:+.1f} um\n")
    f.write(f"EPD {EPD_MM:.3f} mm, f/{fod.fno:.3f}\n")

print(f"\nSaved figures and CSV files to {OUT_DIR}")
