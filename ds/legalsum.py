from ds.hfdataset import HFDataset


class LegalSum(HFDataset):
    ds_name = "legalsum"
    dataset_kwargs = {
        "ds_name": "allenai/multi_lexsum",
        "ds_subset": "v20230518",
        "col_map": {"sources": "text", "summary/long": "summary"},
        "remove_columns": [
            "sources_metadata",
            "summary/short",
            "summary/tiny",
            "id",
            "case_metadata",
        ],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            load_csv=load_csv,
            data_files=data_files,
            **self.dataset_kwargs,
        )
