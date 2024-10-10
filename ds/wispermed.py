from ds.hfdataset import HFDataset


class LaySum(HFDataset):
    ds_name = "wispermed"
    dataset_kwargs = {
        "ds_name": "ishwor-subedi621/slice1000_wispermed",
        "ds_subset": "default",
        "col_map": {"article": "text", "summary": "summary"},
        "remove_columns": ["section_headings", "keywords", "year", "title"],
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
