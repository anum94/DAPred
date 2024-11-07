from ds.hfdataset import HFDataset


class GigaWord(HFDataset):
    ds_name = "gigaword"
    dataset_kwargs = {
        "ds_name": "Harvard/gigaword",
        "ds_subset": "document",
        "col_map": {"document": "text", "summary": "summary"},
        "remove_columns": [],
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
