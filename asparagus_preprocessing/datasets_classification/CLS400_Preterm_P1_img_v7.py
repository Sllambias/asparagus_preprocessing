import logging
import os
import polars as pl
from asparagus_preprocessing.paths import get_raw_labels_path, get_source_path
from asparagus_preprocessing.utils.dataset_registration import register_dataset
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.splitting import non_stratified_split


def load_population(path: str, root: str) -> pl.DataFrame:
    """One row per image of an EHR_extract population, with its absolute path and the GA at birth in days."""
    df = pl.read_parquet(path)
    df = df.select(["CPR_MOR", "CPR_BARN", "GA", "ga_at_scan", "NO_OCR_FILE_PATH"]).drop_nulls("NO_OCR_FILE_PATH")

    # The paths are relative to root. os.path.join keeps paths that are already absolute
    file_paths = [os.path.join(root, file) for file in df["NO_OCR_FILE_PATH"]]
    df = df.with_columns(pl.Series("file_path", file_paths)).drop("NO_OCR_FILE_PATH")

    # Twins share their mother's images. Each image keeps the lowest GA
    n_rows = len(df)
    df = df.sort(["file_path", "GA"]).unique(subset="file_path", keep="first", maintain_order=True)
    logging.warning(f"{path}: {n_rows} image matches, {len(df)} images after keeping one child per image")
    return df


def main(
    path: str = get_source_path(),
    subdir: str = "preterm_P1_img_v7",
    root: str = "/projects/users/data/UCPH/DeepFetal/ultrasound/PNG_PROCESSED_NO_OCR_TMP_DIR",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    # path/subdir is EHR_extract's output folder for preterm_P1_img_v7, which splits train and test on CPR_MOR
    task_name = "CLS400_Preterm_P1_img_v7"
    population_path = os.path.join(path, subdir, f"{subdir}_population")
    trainval_df = load_population(population_path + "_train.parquet", root)
    test_df = load_population(population_path + "_test.parquet", root)

    logging.warning("Splitting the mothers of the train population into train and val")
    mothers = trainval_df["CPR_MOR"].unique().to_list()
    train_mothers, val_mothers = non_stratified_split(mothers, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)
    train_files = trainval_df.filter(pl.col("CPR_MOR").is_in(train_mothers))["file_path"].to_list()
    val_files = trainval_df.filter(pl.col("CPR_MOR").is_in(val_mothers))["file_path"].to_list()
    test_files = test_df["file_path"].to_list()

    # The GA labels for asparagus' TableClsRegDataModule, which binarizes them at the cutoff in the run config
    labels_path = os.path.join(get_raw_labels_path(), task_name, "labels.csv")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    pl.concat([trainval_df, test_df]).write_csv(labels_path)
    logging.warning(f"Saved the labels of {len(trainval_df) + len(test_df)} images to {labels_path}")

    register_dataset(
        task_name=task_name,
        n_classes=2,
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
