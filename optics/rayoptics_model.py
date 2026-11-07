from rayoptics.optical.opticalmodel import OpticalModel

model = OpticalModel()
sm = model.seq_model

# two refracting surfaces for biconvex lens
# approximate radii
# near 30 mm, refractive index ~1.5

sm.add_surface([60.0, 3.0, 1.5])
sm.add_surface([-60.0, 27.0])

model.update_model()

sm.list_model()