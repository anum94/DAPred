from ds.hfdataset import HFDataset


class SAMSum(HFDataset):
    ds_name = "samsum"
    dataset_kwargs = {
        "ds_name": "Samsung/samsum",
        "ds_subset": "samsum",
        "col_map": {"dialogue": "text", "summary": "summary"},
        "remove_columns": ["id"],
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
            **self.dataset_kwargs,
        )
