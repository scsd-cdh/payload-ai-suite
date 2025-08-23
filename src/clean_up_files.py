#!/usr/bin/env python3
"""
Clean up files in the ../data/labeled directory by:
1. Removing duplicate files (keeping one copy)
2. Removing 0-byte files
"""

import os
import hashlib
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.error(f"Error reading {filepath}: {e}")
        return None


def find_and_remove_empty_files(directory, dry_run=True):
    """Find and optionally remove 0-byte files."""
    empty_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Check if file exists and is empty
            if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
                empty_files.append(filepath)
                if not dry_run:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed empty file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing {filepath}: {e}")
    
    return empty_files


def find_and_remove_duplicates(directory, dry_run=True):
    """Find and optionally remove duplicate files, keeping one copy."""
    # Dictionary to store hash -> list of file paths
    hash_to_files = defaultdict(list)
    
    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Skip if it's not a regular file or if it's empty
            if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
                continue
            
            # Compute hash
            file_hash = compute_file_hash(filepath)
            if file_hash:
                hash_to_files[file_hash].append(filepath)
    
    # Process duplicates
    removed_files = []
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            # Sort to ensure consistent behavior (keep the first alphabetically)
            files.sort()
            
            # Keep the first file, remove the rest
            files_to_remove = files[1:]
            
            for filepath in files_to_remove:
                removed_files.append(filepath)
                if not dry_run:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed duplicate: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing {filepath}: {e}")
    
    return removed_files


def main():
    directory = "../data/labeled"
    
    print(f"Scanning for files to clean up in: {directory}")
    print("This may take a moment...")
    print()
    
    # First, find and display what would be removed
    print("=" * 80)
    print("DRY RUN - No files will be deleted")
    print("=" * 80)
    
    # Find empty files
    print("\nFinding empty (0-byte) files...")
    empty_files = find_and_remove_empty_files(directory, dry_run=True)
    
    if empty_files:
        print(f"\nFound {len(empty_files)} empty files:")
        for f in sorted(empty_files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(empty_files) > 10:
            print(f"  ... and {len(empty_files) - 10} more")
    else:
        print("No empty files found!")
    
    # Find duplicate files
    print("\nFinding duplicate files...")
    duplicate_files = find_and_remove_duplicates(directory, dry_run=True)
    
    if duplicate_files:
        print(f"\nFound {len(duplicate_files)} duplicate files that would be removed:")
        for f in sorted(duplicate_files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(duplicate_files) > 10:
            print(f"  ... and {len(duplicate_files) - 10} more")
    else:
        print("No duplicate files found!")
    
    # Calculate space that would be freed
    total_space = 0
    for f in empty_files + duplicate_files:
        try:
            total_space += os.path.getsize(f)
        except:
            pass
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"- Empty files to remove: {len(empty_files)}")
    print(f"- Duplicate files to remove: {len(duplicate_files)}")
    print(f"- Total files to remove: {len(empty_files) + len(duplicate_files)}")
    if total_space > 0:
        print(f"- Space to be freed: {total_space:,} bytes ({total_space / (1024*1024):.2f} MB)")
    
    # Ask user for confirmation
    if empty_files or duplicate_files:
        print("\n" + "=" * 80)
        response = input("\nDo you want to proceed with deletion? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print("\nProceeding with deletion...")
            
            # Remove empty files
            if empty_files:
                print("\nRemoving empty files...")
                find_and_remove_empty_files(directory, dry_run=False)
            
            # Remove duplicate files
            if duplicate_files:
                print("\nRemoving duplicate files...")
                find_and_remove_duplicates(directory, dry_run=False)
            
            print("\nCleanup completed!")
        else:
            print("\nOperation cancelled. No files were deleted.")
    else:
        print("\nNo files to clean up!")


if __name__ == "__main__":
    main()