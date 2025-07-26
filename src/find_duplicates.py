#!/usr/bin/env python3
"""
Find duplicate files in the data/labeled directory by comparing checksums.
"""

import os
import hashlib
from collections import defaultdict


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def find_duplicates(directory):
    """Find duplicate files in the given directory."""
    # Dictionary to store hash -> list of file paths
    hash_to_files = defaultdict(list)

    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)

            # Skip if it's not a regular file
            if not os.path.isfile(filepath):
                continue

            # Compute hash
            file_hash = compute_file_hash(filepath)
            if file_hash:
                hash_to_files[file_hash].append(filepath)

    # Filter out hashes that only have one file (no duplicates)
    duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}

    return duplicates


def main():
    directory = "/home/kp/dev/space/payload-ai-suite/data/labeled"

    print(f"Scanning for duplicate files in: {directory}")
    print("This may take a moment...")

    # Get all file hashes first
    hash_to_files = defaultdict(list)
    total_files_scanned = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)

            # Skip if it's not a regular file
            if not os.path.isfile(filepath):
                continue

            total_files_scanned += 1

            # Compute hash
            file_hash = compute_file_hash(filepath)
            if file_hash:
                hash_to_files[file_hash].append(filepath)

    # Filter out hashes that only have one file (no duplicates)
    duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}

    if not duplicates:
        print("\nNo duplicate files found!")
        return

    # Count total duplicate files
    total_duplicate_files = sum(len(files) for files in duplicates.values())
    total_duplicate_groups = len(duplicates)

    print(f"\nFound {total_duplicate_groups} groups of duplicate files")
    print(f"Total duplicate files: {total_duplicate_files}")
    print("\nDuplicate file groups:")
    print("=" * 80)

    for idx, (file_hash, files) in enumerate(duplicates.items(), 1):
        print(f"\nGroup {idx} (Hash: {file_hash[:16]}...):")

        # Get file size
        try:
            file_size = os.path.getsize(files[0])
            print(f"File size: {file_size:,} bytes")
        except:
            pass

        # Group files by 'yes' or 'no' folder
        yes_files = [f for f in files if '/yes/' in f]
        no_files = [f for f in files if '/no/' in f]

        if yes_files:
            print(f"  In 'yes' folder ({len(yes_files)} files):")
            for f in sorted(yes_files):
                print(f"    - {os.path.basename(f)}")

        if no_files:
            print(f"  In 'no' folder ({len(no_files)} files):")
            for f in sorted(no_files):
                print(f"    - {os.path.basename(f)}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"- Total files scanned: {total_files_scanned}")
    print(f"- Unique files: {len(hash_to_files) - len(duplicates)}")
    print(f"- Duplicate groups: {total_duplicate_groups}")
    print(f"- Total duplicate files: {total_duplicate_files}")

    # Calculate wasted space
    wasted_space = 0
    for files in duplicates.values():
        try:
            file_size = os.path.getsize(files[0])
            # Each duplicate after the first is wasted space
            wasted_space += file_size * (len(files) - 1)
        except:
            pass

    if wasted_space > 0:
        print(f"- Wasted space from duplicates: {wasted_space:,} bytes ({wasted_space / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()