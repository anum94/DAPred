from ds.hfdataset import HFDataset


class DialogSum(HFDataset):
    ds_name = "dialogsum"
    dataset_kwargs = {
        "ds_name": "knkarthick/dialogsum",
        "ds_subset": "default",
        "col_map": {"dialogue": "text", "summary": "summary"},
        "remove_columns": ["id", "topic"],
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
