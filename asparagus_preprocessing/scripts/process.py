import logging
from inspect import signature

from asparagus_preprocessing.utils.parser import asparagus_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = asparagus_parser
    parser.add_argument(
        "--dataset",
        help="Name of the dataset to preprocess. Should be of format: PT001_NAME or SEG001_NAME etc.",
        required=True,
    )
    parser.add_argument(
        "--task_name",
        help="Override the processed dataset name for dataset modules that support it.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        help="Select modalities for dataset modules that support modality filtering.",
    )
    args = parser.parse_args()
    module = find_module(args.dataset)

    module_kwargs = {}
    supported_params = signature(module.main).parameters
    for name in ["task_name", "modalities"]:
        value = getattr(args, name)
        if value is not None:
            if name not in supported_params:
                raise ValueError(f"{args.dataset} does not support --{name}")
            module_kwargs[name] = value

    module.main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
        **module_kwargs,
    )


def find_module(substring: str):
    import asparagus_preprocessing
    import importlib
    import os

    root = asparagus_preprocessing.__path__[0]
    for dir, subdirs, files in os.walk(root):
        for file in files:
            if (file.startswith(substring + "_") and file.endswith(".py")) or (file == substring + ".py"):
                module_path = f"asparagus_preprocessing.{dir.replace(root + '/', '')}.{file[:-3]}"
                return importlib.import_module(module_path)
    raise ValueError(f"Module with substring {substring} not found")


if __name__ == "__main__":
    main()
