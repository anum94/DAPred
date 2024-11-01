from ds.hfdataset import HFDataset


class NewsRoom(HFDataset):
    ds_name = "samsum"
    dataset_kwargs = {
        "ds_name": "lil-lab/newsroom",
        "ds_subset": "default",
        "col_map": {"text": "text", "summary": "summary"},
        "remove_columns": [
            "title",
            "url",
            "date",
            "density",
            "coverage",
            "compression",
            "density_bin",
            "coverage_bin",
            "compression_bin",
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
            **self.dataset_kwargs,
        )
