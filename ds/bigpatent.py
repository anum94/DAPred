from ds.hfdataset import HFDataset


class BigPatent(HFDataset):
    ds_name = "bigpatent"
    dataset_kwargs = {
        "ds_name": "satpalsr/bigpatent-test",
        "ds_subset": "default",
        "col_map": {"description": "text", "abstract": "summary"},
        "remove_columns": ["word_length"],
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
