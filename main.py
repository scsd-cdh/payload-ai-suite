"""CLI entry point for tools access.
"""
import argparse
import model

if __name__ == "__main__":
    """Parses command-line arguments and executes the appropriate functionality.

    This script serves as the entry point for the Payload AI Software Suite. It
    allows users to run a model or display a message if no arguments are provided.

    Command-line Arguments:
        -r, --run-model (bool): Whether to run the model.

    Raises:
        SystemExit: If invalid arguments are provided.
    """
    parser = argparse.ArgumentParser(
        prog='Payload AI Software Suite',
        description='Remote sensing mission core tools for wildfire image classification and training data retrieval'
    )
    # VGG16 and ResNet50 supported for now
    parser.add_argument('-r', '--run-model', required=False, type=bool)

    args = parser.parse_args()
    if args.run_model:
        model.run()
    else:
        print("No args sent to CLI")
