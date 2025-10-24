"""CLI entry point for tools suite access. As needed, add appropriate argument options as the project grows.
"""
import argparse
import logging
import model
import cloud as cloud
from fetch import (
    nasa_firms_api,
    setup_auth,
    batch_data_downloader_selenium,
    retrieve_eonet_cross_reference,
    copernicus_query,
    convert_sen2fire_labeled
)
from postprocess import (
    test_ccds
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """
    Parses command-line arguments and executes the appropriate functionality.

    This script serves as the entry point for the Payload AI Software Suite. It
    allows users to run a model or execute specific data-fetching functions.

    Command-line Arguments:
        --run-model: Run the model.
        --nasa-firms: Fetch data from NASA FIRMS API.
        --setup-auth: Set up OAuth2 authentication for Copernicus.
        --batch-download: Download images using Selenium.
        --eonet-crossref: Fetch wildfire data from the EONET API.
        --copernicus-query: Query remote satellite data from Copernicus.
        --coordinates: Specify coordinates for the query in the format: LON LAT.
        --time-range: Time range for the query in the format: FROM TO
                      (e.g., '2023-01-01T00:00:00Z 2023-01-03T23:59:59Z').
        --cloud-mask: Export OmniCloudMask models to ONNX and test on labeled data.

    Raises:
        SystemExit: If invalid arguments are provided.
    """
    parser = argparse.ArgumentParser(
        prog='Payload AI Software Suite',
        description='Remote sensing mission core tools for wildfire image classification and data retrieval'
    )
    parser.add_argument('--run-model', required=False, action='store_true', help="Run the model")
    parser.add_argument('--nasa-firms', required=False, action='store_true', help="Fetch data from NASA FIRMS API")
    parser.add_argument('--setup-auth', required=False, action='store_true', help="Set up OAuth2 authentication for Copernicus")
    parser.add_argument('--batch-download', required=False, action='store_true', help="Download images using Selenium")
    parser.add_argument('--eonet-crossref', required=False, action='store_true', help="Fetch wildfire data from the EONET API")
    parser.add_argument('--copernicus-query', required=False, action='store_true', help="Query Sentinel data from Copernicus")
    parser.add_argument('--coordinates', required=False, nargs=2, type=float, metavar=('LON', 'LAT'),
                        help="Specify coordinates for the query in the format: LON LAT")
    parser.add_argument('--time-range', required=False, nargs=2, metavar=('FROM', 'TO'),
                        help="Time range for the query in the format: FROM TO (e.g., '2023-01-01T00:00:00Z 2023-01-03T23:59:59Z')")
    parser.add_argument('--use-nir', required=False, action='store_true', help="Enable 4-channel RGB-NIR input")
    parser.add_argument('--multimodal-qc', required=False, action='store_true', help="Run multimodal quality control check")
    parser.add_argument('--qc-path', required=False, type=str, help="Path to folder for QC processing (e.g., 'sen2fire/to_qc' or 'eonet_fire_events/to_process')")
    parser.add_argument('--use-gcs', required=False, action='store_true', help="Stream training data from Google Cloud Storage")
    parser.add_argument('--process-sen2fire', required=False, action='store_true', help="Convert Sen2Fire dataset to a state to be processed by the pipeline.")
    parser.add_argument('--cloud-mask', required=False, action='store_true', help="Export OmniCloudMask models to ONNX and test on labeled data")
    parser.add_argument('--upload-labeled', required=False, action='store_true', help="Upload labeled data to GCS after running cleanup")
    parser.add_argument('--download-labeled', required=False, action='store_true', help="Download labeled data from GCS to local filesystem")
    parser.add_argument('--use-mixed-res', required=False, action='store_true', help="Enable mixed resolution operations during training")
    parser.add_argument('--epochs', required=False, type=int, default=12, help="Number of epochs for training")
    parser.add_argument('--fusion-technique', required=False, type=str, default='enhanced_red', choices=['enhanced_red', 'hsv', 'none'], help="RGB-NIR fusion technique")
    parser.add_argument('--fusion-alpha', required=False, type=float, default=0.5, help="Alpha parameter for enhanced_red fusion (0-1)")
    parser.add_argument('--degrade-gsd', required=False, action='store_true', help="Degrade imagery to CubeSat GSD (~85m from 10m Sentinel-2)")
    parser.add_argument("--compress-image", required=False, type=str, help="Path to the input .NEF image")

    args = parser.parse_args()
    if args.run_model:
        experiment_id = model.train(use_nir=args.use_nir, use_gcs=args.use_gcs,
                                    use_mixed_res=args.use_mixed_res, epochs=args.epochs,
                                    fusion_technique=args.fusion_technique,
                                    fusion_alpha=args.fusion_alpha,
                                    degrade_gsd=args.degrade_gsd)
        print(f"\nTraining complete! Experiment ID: {experiment_id}")
    elif args.nasa_firms:
        nasa_firms_api()
    elif args.setup_auth:
        setup_auth()
    elif args.batch_download:
        batch_data_downloader_selenium()
    elif args.eonet_crossref:
        retrieve_eonet_cross_reference()
    elif args.copernicus_query:
        copernicus_query(use_gcs=args.use_gcs)
    elif args.multimodal_qc:
        import mlops
        mlops.run_multimodal_qc(use_gcs=args.use_gcs, input_path=args.qc_path)
    elif args.process_sen2fire:
        convert_sen2fire_labeled(use_nir=args.use_nir)
    elif args.cloud_mask:
        cloud.main()
    elif args.upload_labeled:
        import mlops
        mlops.upload_labeled_to_gcs()
    elif args.download_labeled:
        import mlops
        mlops.download_labeled_from_gcs()
    elif args.compress_image:
        test_ccds(args.compress_image)
    else:
        logger.error("No valid arguments provided. Use -h for help.")
