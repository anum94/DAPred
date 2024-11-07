from ds.hfdataset import HFDataset


class XlSum(HFDataset):
    ds_name = "xlsum"
    dataset_kwargs = {
        "ds_name": "csebuetnlp/xlsum",
        "ds_subset": "english",
        "col_map": {"text": "text", "summary": "summary"},
        "remove_columns": ["id", "url", "title"],
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
