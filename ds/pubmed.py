from ds.hfdataset import HFDataset


class Pubmed(HFDataset):
    ds_name = "pubmed"
    dataset_kwargs = {
        "ds_name": "ishwor-subedi621/slice1000_pubmed",
        "ds_subset": "default",
        "col_map": {"text": "text", "summary": "summary"},
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
