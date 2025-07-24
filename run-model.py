# Run from your main script or notebook
import model

# Train with mixed resolution operations enabled
model.train(validate=True, epochs=15, use_nir=False, use_gcs=False, use_mixed_res=True)