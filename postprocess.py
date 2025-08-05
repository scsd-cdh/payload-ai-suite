import os
import rawpy
import imageio

def conversion():
  #Initial variables
  input_folder="images"
  output_folder="output"
  quality=80

  #Make output folder for AVIFs if it doesn't exist already
  os.makedirs(output_folder, exist_ok=True)

  #Go through every raw image
  for filename in os.listdir(input_folder):
    #Can add more extensions later if there's others
    if filename.lower().endswith(".nef"):
      input_path = os.path.join(input_folder, filename)
      output_filename = os.path.splitext(filename)[0] + ".avif"
      output_path = os.path.join(output_folder, output_filename)

      try:
        with rawpy.imread(input_path) as raw:
          rgb = raw.postprocess(
              use_camera_wb=True,
              gamma=(2.2, 4.5),
              output_bps=8,
              no_auto_bright=True
          )
          imageio.imwrite(output_path, rgb, format="avif", quality=quality)
          print(f"Converted: {filename}")
      except Exception as e:
        print(f"Failed to convert {filename}: {e}")
