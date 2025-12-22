from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class PreprocessingConfig:
    """
    Normalization:
        Supply a list of normalization operations with one entry per modality. E.g. ["volume_wise_znorm", "no_norm"]
        if the first modality should be normalized and the second should remain untouched
        - as is often the case if segmentations are used as training data
    Output dimensions:
        By default this will be defined by the target spacing. This means preprocessed data will often vary in
        height, width and depth dimensions.
        If uniform dimensions are desired set target_spacing to None and use target_size instead.
        Optionally with keep_aspect_ratio_when_using_target_size set to true if aspect ratio is important.
        This will introduce black bars in some dimensions to achieve specific dimensions without compromising aspect ratio.
    """

    normalization_operation: List
    target_spacing: Optional[List]
    background_pixel_value: int = 0
    crop_to_nonzero: bool = True
    keep_aspect_ratio_when_using_target_size: bool = False
    image_properties: Optional[dict] = field(default_factory=dict)
    intensities: Optional[List] = None
    target_orientation: Optional[str] = "RAS"
    target_size: Optional[List] = None
    min_slices: int = 0
    remove_nans: bool = True


@dataclass
class SavingConfig:
    save_as_tensor: bool
    tensor_dtype: str
    bidsify: bool
    save_dset_metadata: bool
    save_file_metadata: bool


@dataclass
class DatasetConfig:
    df_columns: list
    task_name: str
    n_classes: int
    n_modalities: int
    in_extensions: str
    patterns_exclusion: list
    patterns_DWI: list
    patterns_PET: list
    patterns_perfusion: list
    patterns_m0: list
    patterns_bidsify: list
    split: str
