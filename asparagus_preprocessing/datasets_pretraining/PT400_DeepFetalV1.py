import logging
import os
import polars as pl
from asparagus_preprocessing.utils.dataset_registration import register_dataset
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.splitting import non_stratified_split


def main(
    path: str = "/projects/users/data/UCPH/DeepFetal/ultrasound/tables/database_snapshots/all_images_2026-11-11.csv",
    root: str = "/projects/users/data/UCPH/DeepFetal/ultrasound/PNG_processed_no_OCR",
    holdout_csv: str = "/projects/users/data/UCPH/DeepFetal/projects/common/splits/split_V4-1/test_split_0.15_2026-11-16.csv",
    holdout_key: str = "CPR_MOR",
    subdir: str = "",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    logging.warning("Loading DF")
    df = pl.read_csv(path)
    df = df.select(["file_hash", "no_ocr_preprocessed_file_path"]).drop_nulls()

    # The paths are relative to root. os.path.join keeps paths that are already absolute
    file_paths = [os.path.join(root, file) for file in df["no_ocr_preprocessed_file_path"]]
    df = df.with_columns(pl.Series("file_path", file_paths))
    # is_in copies its list once per chunk
    df = df.rechunk()

    logging.warning("Holding out the subjects in the holdout CSV as the test set")
    holdout = pl.read_csv(holdout_csv)[holdout_key].unique().to_list()
    test_df = df.filter(pl.col("file_hash").is_in(holdout))
    trainval_df = df.filter(~pl.col("file_hash").is_in(holdout))
    logging.warning(
        f"{test_df['file_hash'].n_unique()} of the {len(holdout)} held-out subjects have images, "
        f"{trainval_df['file_hash'].n_unique()} subjects remain for train and val"
    )

    logging.warning("Splitting the remaining subjects into train and val")
    subjects = trainval_df["file_hash"].unique().to_list()
    train_subjects, val_subjects = non_stratified_split(subjects, train_ratio=0.99, val_ratio=0.01, test_ratio=0.0)

    train_files = trainval_df.filter(pl.col("file_hash").is_in(train_subjects))["file_path"].to_list()
    val_files = trainval_df.filter(pl.col("file_hash").is_in(val_subjects))["file_path"].to_list()
    test_files = test_df["file_path"].to_list()

    logging.warning("Saving")
    register_dataset(
        task_name="PT400_DeepFetalV1",
        n_classes=1,
        n_modalities=1,
        train=train_files,
        val=val_files,
        test=test_files,
        n_folds=1,
    )


if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )
