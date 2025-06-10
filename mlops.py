import os
import shutil
import glob

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def multimodal_qc(file_path):
    """
    Performs a multimodal quality control check on an image file using the Gemini 2.0 Flash model.

    This function uploads an image file to the Gemini API, asks the model to determine if there's a fire in the image,
    and then categorizes the image as 'fire' or 'no fire' based on the model's response.
    The processed image is then copied to a corresponding labeled directory.

    Args:
        file_path (str): The path to the image file to be processed.

    Returns:
        str: "fire" if the model detects a fire, "no fire" otherwise.
    """
    print(f"Processing file: {file_path}")
    file_input = client.files.upload(file=file_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[file_input, "Is there a fire in this image? Respond with 'yes' if there is a fire, or 'no' if there is no fire."],
    )
    raw_response_text = response.text.strip().lower()
    print(f"Raw response for {file_path}: {raw_response_text}")

    if "yes" in raw_response_text and "no" not in raw_response_text:
        binary_output = "fire"
    elif "no" in raw_response_text:
        binary_output = "no fire"
    else:
        print(f"Warning: Ambiguous response for {file_path}: {raw_response_text}. Defaulting to 'no fire'.")
        binary_output = "no fire"

    print(f"Binary output for {file_path}: {binary_output}")
    output_dir = ""
    if binary_output == "fire":
        output_dir = "./data/labeled/yes"
    else:
        output_dir = "./data/labeled/no"

    file_name = os.path.basename(file_path)
    destination_path = os.path.join(output_dir, file_name)
    shutil.copy(file_path, destination_path)
    print(f"File saved to: {destination_path}")

    return binary_output

def run_multimodal_qc():
    """
    Orchestrates the multimodal quality control process for a batch of image files.

    This function identifies image files in a predefined directory
    ('data/eonet_fire_events/to_process/') and processes each of them
    using the `multimodal_qc` function.
    """
    image_files = glob.glob("data/eonet_fire_events/to_process/*")

    # Process the first 10 files
    for i, file_path in enumerate(image_files):
        multimodal_qc(file_path)
