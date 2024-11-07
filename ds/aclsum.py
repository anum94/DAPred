from ds.hfdataset import HFDataset


class AclSum(HFDataset):
    ds_name = "aclsum"
    dataset_kwargs = {
        "ds_name": "sobamchan/aclsum",
        "ds_subset": "abstractive",
        "col_map": {"document": "text", "summary": "summary"},
        "remove_columns": ["challenge", "approach", "outcome"],
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
