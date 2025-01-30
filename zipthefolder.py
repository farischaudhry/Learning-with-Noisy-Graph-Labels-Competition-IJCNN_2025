import tarfile
import os
import argparse

def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a folder into a .tar.gz archive.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder to compress")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .tar.gz file")
    args = parser.parse_args()

    gzip_folder(args.input_folder, args.output_file)