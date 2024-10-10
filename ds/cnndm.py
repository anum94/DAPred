from ds.hfdataset import HFDataset


class CNNDailyMail(HFDataset):
    ds_name = "cnndm"
    dataset_kwargs = {
        "ds_name": "abisee/cnn_dailymail",
        "ds_subset": "1.0.0",  # other options: "2.0.0" and  "3.0.0"
        "col_map": {"article": "text", "highlights": "summary"},
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
