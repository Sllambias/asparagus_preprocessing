def get_image_output_paths(files, source_dir, target_dir, file_suffix=[".nii", ".nii.gz"]):
    """Generates image file paths files and metadata paths with the input structure in the output directory"""
    files_out = []
    for f in files:
        file_out = f.replace(source_dir, target_dir)
        # Replace any matching file suffix with the image suffix
        for suffix in file_suffix:
            if file_out.endswith(suffix):
                file_out = file_out[: -len(suffix)]
                break
        files_out.append(file_out)

    return files_out
