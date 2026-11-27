"""
Fit the two-element stand-in for the Schneider TOPAZ 2.0/30 C (Issue #58).

You only need to run this if you want to re-derive the lens shape that
rayoptics_model.py uses (LENS_PRESCRIPTION). It takes about a minute.

Why a fit is needed
    The real TOPAZ prescription is not public. The datasheet gives first-order
    data (focal length, focal distances, pupil positions, length). A simple
    two-lens model cannot match all of them exactly, so we fit a balance of:
      * EFL and S'F' (back focal distance)  -> weighted x100, matched exactly
      * SF, SEP, S'AP, pupil magnification  -> best fit
      * on-axis and edge spot size at 525 nm -> keeps the lens reasonably sharp
      * edge thickness >= 1 mm for each lens -> physically buildable
    Total length (Sum d = 29.17 mm) and f/2 are enforced by construction.

Run:  python optics/fit_equivalent_lens.py
"""

import numpy as np
from scipy.optimize import least_squares
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec
from rayoptics.raytr.trace import trace_base

# TOPAZ 2.0/30 C datasheet values
EFL, SPF, SF, SEP, SUM_D, SPAP, PUPIL_MAG = 29.99, 13.96, -4.91, 17.05, 29.17, -27.0, 1.37
F_NUMBER = 2.0
GLASS = ("N-BK7", "Schott")          # ASSUMPTION: real glasses unknown
FIT_ANGLES = [0.0, 10.64, 14.87]     # on-axis, sensor edge, sensor corner (deg)
MIN_EDGE_MM = 1.0                    # ASSUMPTION: minimum lens edge thickness


def build(p):
    """p = [c1, c2, c3, c4, gap lens1->stop, t1, t2, focus shift]."""
    c1, c2, c3, c4, g1, t1, t2, dz = p
    g2 = SUM_D - t1 - t2 - g1
    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    osp["pupil"] = PupilSpec(osp, key=["object", "epd"], value=EFL / F_NUMBER)
    osp["fov"] = FieldSpec(osp, key=["object", "angle"], flds=FIT_ANGLES)
    osp["wvls"] = WvlSpec([(525.0, 1.0)], ref_wl=0)
    opm.radius_mode = False
    sm.gaps[0].thi = 1e10
    sm.add_surface([c1, t1, *GLASS])
    sm.add_surface([c2, g1])
    sm.add_surface([0.0, g2])
    sm.set_stop()
    sm.add_surface([c3, t2, *GLASS])
    sm.add_surface([c4, SPF + dz])
    opm.update_model()
    return opm


GRID = [(x, y) for x in np.linspace(-1, 1, 7) for y in np.linspace(-1, 1, 7)
        if x * x + y * y <= 1.0001]


def rms_spot(opm, fi):
    fld = opm["optical_spec"]["fov"].fields[fi]
    pts = []
    for g in GRID:
        try:
            pts.append(trace_base(opm, list(g), fld, 525.0)[0][-1][0][:2])
        except Exception:
            return 5.0                      # big penalty if a ray fails
    pts = np.array(pts)
    return np.sqrt(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1)))


def sag(c, h):
    if abs(c) < 1e-12:
        return 0.0
    v = 1 / c**2 - h * h
    return np.sign(c) * abs(1 / c) if v < 0 else 1 / c - np.sign(c) * np.sqrt(v)


def edge_thicknesses(opm, p):
    ifc = opm["seq_model"].ifcs
    out = []
    for a, b, ca, cb, t in [(1, 2, p[0], p[1], p[5]), (4, 5, p[2], p[3], p[6])]:
        h = 1.05 * max(ifc[a].surface_od(), ifc[b].surface_od())  # 5% margin
        out.append(t - sag(ca, h) + sag(cb, h))
    return out


def exit_pupil_from_last_vertex(opm):
    """S'AP computed directly from the paraxial chief ray after the last lens.
    (fod.exp_dist shifts when the sensor is moved, so it is not used here.)"""
    pr = opm["parax_model"].pr
    k = len(opm["seq_model"].ifcs) - 2          # last lens surface
    return -pr[k][0] / pr[k][1]


def residuals(p):
    try:
        opm = build(p)
    except Exception:
        return np.full(10, 1e3)
    f = opm["analysis_results"]["parax_data"].fod
    e = edge_thicknesses(opm, p)
    return np.array([
        100 * (f.efl - EFL),
        100 * (f.bfl - SPF),
        f.ffl - SF,
        f.enp_dist - SEP,
        0.3 * (exit_pupil_from_last_vertex(opm) - SPAP),
        3 * (f.exp_radius / f.enp_radius - PUPIL_MAG),
        100 * rms_spot(opm, 0),             # mm -> weight: 10 um RMS ~ 1 mm error
        100 * rms_spot(opm, 1),
        30 * max(0.0, MIN_EDGE_MM - e[0]),
        30 * max(0.0, MIN_EDGE_MM - e[1]),
    ])


if __name__ == "__main__":
    lo = [-0.2] * 4 + [1.0, 1.5, 1.5, -2.0]
    hi = [0.2] * 4 + [20.0, 9.0, 9.0, 1.0]
    rng = np.random.default_rng(7)
    starts = [[1/55, -1/55, 1/33, -1/33, 12, 3.5, 3.5, -0.5]] + [
        list(rng.uniform(-0.05, 0.06, 4)) +
        [rng.uniform(4, 13), rng.uniform(3, 6), rng.uniform(3, 6), -0.4]
        for _ in range(15)]
    best = None
    for i, x0 in enumerate(starts):
        s = least_squares(residuals, x0=x0, bounds=(lo, hi), max_nfev=200)
        if SUM_D - s.x[4] - s.x[5] - s.x[6] < 0.5:
            continue                        # gap after the stop too small
        if best is None or s.cost < best.cost:
            best = s
        print(f"start {i + 1}/{len(starts)}  cost {s.cost:.2f}")
    best = least_squares(residuals, x0=best.x, bounds=(lo, hi), max_nfev=800)

    p = best.x
    opm = build(p)
    f = opm["analysis_results"]["parax_data"].fod
    print("\nPaste into rayoptics_model.py as LENS_PRESCRIPTION:")
    print("LENS_PRESCRIPTION = {")
    print(f'    "c1": {p[0]:.8f}, "c2": {p[1]:.8f},   # lens 1 curvatures (1/mm)')
    print(f'    "c3": {p[2]:.8f}, "c4": {p[3]:.8f},   # lens 2 curvatures (1/mm)')
    print(f'    "gap_lens1_stop": {p[4]:.6f},')
    print(f'    "t1": {p[5]:.6f}, "t2": {p[6]:.6f},')
    print("}")
    print(f"\nEFL {f.efl:.3f}  S'F' {f.bfl:.3f}  SF {f.ffl:.3f}  SEP {f.enp_dist:.3f}  "
          f"S'AP {exit_pupil_from_last_vertex(opm):.3f}  beta'P {f.exp_radius / f.enp_radius:.3f}")
    print("Edge thickness (mm):", np.round(edge_thicknesses(opm, p), 2))
