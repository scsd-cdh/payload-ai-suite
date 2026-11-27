
import numpy as np
import matplotlib.pyplot as plt

# Zetane optical system parameters

# Lens: Schneider TOPAZ 2.0/30 C (from the lens documentation / datasheet)
FOCAL_LENGTH_MM = 30.0      # nominal focal length (datasheet effective value: 29.99 mm)
F_NUMBER = 2.0              # lens at its widest aperture (range F/2 ... F/16)

# Camera sensor (from the camera documentation)
PIXEL_PITCH_MM = 0.0055     # 5.5 um x 5.5 um pixels
SENSOR_RESOLUTION = 2048    # 2048 x 2048 pixels (square sensor, 1:1)

# Orbit: 400 km nominal, 450 km worst case (largest GSD)
ORBIT_ALTITUDES_KM = {"Nominal": 400.0, "Worst case": 450.0}

# Basic optical calculations
sensor_width = SENSOR_RESOLUTION * PIXEL_PITCH_MM
aperture_diameter = FOCAL_LENGTH_MM / F_NUMBER

fov_rad = 2 * np.arctan(sensor_width / (2 * FOCAL_LENGTH_MM))
fov_deg = np.degrees(fov_rad)

print("ZETANE OPTICAL SYSTEM")
print("---------------------")
print(f"Sensor width: {sensor_width:.3f} mm")
print(f"Aperture diameter: {aperture_diameter:.2f} mm")
print(f"Field of view: {fov_deg:.2f} degrees")

for case, altitude_km in ORBIT_ALTITUDES_KM.items():
    gsd_m = altitude_km * 1000 * PIXEL_PITCH_MM / FOCAL_LENGTH_MM
    swath_km = 2 * altitude_km * np.tan(fov_rad / 2)
    print(f"\n{case} ({altitude_km:.0f} km)")
    print(f"  GSD: {gsd_m:.2f} m/pixel")
    print(f"  Ground swath: {swath_km:.2f} km")

# Simplified ideal thin-lens ray diagram
fig, ax = plt.subplots(figsize=(10, 5))

for ray_height in np.linspace(-aperture_diameter / 2, aperture_diameter / 2, 7):
    ax.plot([-25, 0], [ray_height, ray_height], color="tab:blue")
    ax.plot([0, FOCAL_LENGTH_MM], [ray_height, 0], color="tab:orange")

ax.axvline(0, color="black", linewidth=2, label="Ideal lens")
ax.axvline(FOCAL_LENGTH_MM, color="green", linestyle="--", label="Sensor")
ax.axhline(0, color="gray", linewidth=0.5)

ax.set_title("Zetane - Simplified 30 mm Lens Ray Diagram")
ax.set_xlabel("Optical axis (mm)")
ax.set_ylabel("Ray height (mm)")
ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
