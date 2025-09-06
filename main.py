"""CLI entry point for tools suite access. As needed, add appropriate argument options as the project grows.
"""
import argparse
import model
import mlops
import cloud
from fetch import (
    nasa_firms_api,
    setup_auth,
    batch_data_downloader_selenium,
    retrieve_eonet_cross_reference,
    copernicus_sentiel_query,
    enmap_fetch_for_events,
)

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
        --copernicus-query: Query Sentinel data from Copernicus.
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
    parser.add_argument('--use-gcs', required=False, action='store_true', help="Stream training data from Google Cloud Storage")
    parser.add_argument('--cloud-mask', required=False, action='store_true', help="Export OmniCloudMask models to ONNX and test on labeled data")
    parser.add_argument("--enmap-for-events", action="store_true", help="Find and download EnMAP L2A near each event")
    parser.add_argument("--events-json", type=str, default="events/categories.json", help="Path to the JSON file containing event records (default: events/categories.json)")
    parser.add_argument("--date-range", type=str, default="2024-01-01/2024-12-31", help="Date range to search for EnMAP products in the format 'YYYY-MM-DD/YYYY-MM-DD'")
    parser.add_argument("--radius-km", type=float, default=10.0, help="Radius in kilometers around each event's coordinates to use when querying for imagery "
    "(default: 10.0 km)")
    parser.add_argument("--per-event-limit", type=int, default=1, help="Maximum number of EnMAP images to download per event (default: 1)")
    parser.add_argument("--out-enmap", type=str, default="data/enmap", help="Output directory where downloaded EnMAP images will be stored (default: data/enmap)")

    args = parser.parse_args()
    if args.run_model:
        model.train(use_nir=args.use_nir, use_gcs=args.use_gcs)
    elif args.nasa_firms:
        nasa_firms_api()
    elif args.setup_auth:
        setup_auth()
    elif args.batch_download:
        batch_data_downloader_selenium()
    elif args.eonet_crossref:
        retrieve_eonet_cross_reference()
    elif args.copernicus_query:
        copernicus_sentiel_query(use_gcs=args.use_gcs)
    elif args.multimodal_qc:
        mlops.run_multimodal_qc(use_gcs=args.use_gcs)
    elif args.cloud_mask:
        cloud.main()
    elif args.enmap_for_events:
        result = enmap_fetch_for_events(
            events_json_path=args.events_json,
            out_root=args.out_enmap,
            default_date_range=args.date_range,
            radius_km=args.radius_km,
            per_event_limit=args.per_event_limit
        )
        print("[ENMAP] Summary:",
              f"{len(result['downloaded'])} downloads, {len(result['failures'])} failures")
    else:
        print("No valid arguments provided. Use -h for help.")

