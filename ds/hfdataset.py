import logging
from typing import Mapping, Sequence, Union

from datasets import Dataset, DatasetDict, load_dataset


class HFDataset:
    ds: DatasetDict
    ds_name: str

    def __init__(
        self,
        ds_name: str,
        ds_subset: str,
        col_map: dict,
        remove_columns: list,
        preview: bool,
        preview_size: int = 4,
        min_input_size: int = 0,
        samples: Union[int, str] = "max",
        load_csv: bool = False,
        data_files: Union[
            dict, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None
        ] = None,
    ) -> None:
        if load_csv:
            data = load_dataset("csv", data_files=data_files)
        else:
            if ds_name == "lil-lab/newsroom":
                data = load_dataset(
                    ds_name,
                    data_dir=".cache/huggingface/datasets/newsroom",
                    trust_remote_code=True,
                )
            else:
                data = load_dataset(ds_name, ds_subset, trust_remote_code=True)

            #print("DATASET_NAME:", ds_name)

            if ds_name == "allenai/multi_lexsum":
                data = self.combine_document_field(dataset=data)
            elif ds_name == "sobamchan/aclsum":
                data = self.combine_columns(dataset=data)

        if preview:
            for k in data.keys():
                data[k] = data[k].select(range(preview_size))

        elif samples != "max":
            data["train"] = data["train"].select(
                range(min(len(data["train"]), samples))
            )

        self.ds = self.preprocess(
            data, col_map, min_input_size, remove_columns=remove_columns
        )

    def get_split(self, key: str) -> Dataset:
        return self.ds[key]

    def scrolls_preprocess(
        self,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
        remove_columns: list,
    ) -> DatasetDict:
        # conditions for admitting data into the training:
        # 1) Text (x) is twice as long as summary (y) and less than 1000 times longer.
        # 2) Summary is not a verbatim part of the text.
        # 3) The text document has a minimum length (min_input_size).
        def mask(x, y):
            return (
                2 * len(y) < len(x) < 1000 * len(y)
                and y not in x
                and len(x) >= min_input_size
            )

        def none_data_filter(example):
            return example["text"] is not None and example["summary"] is not None

        def fn(batch: dict):
            res = {"text": [], "summary": []}
            z = zip(batch["text"], batch["summary"])
            # apply the logical inverse of `mask` to obtain admissible documents.
            valid = list(filter(lambda x: mask(x[0], x[1]), z))
            # print(valid)
            res["text"] = [valid[idx][0] for idx in range(len(valid))]
            res["summary"] = [valid[idx][1] for idx in range(len(valid))]
            return res

        logging.info("Preprocessing dataset")
        data = data.rename_columns(col_map)

        # save_test = data["test"]
        #print(data)

        data = data.filter(none_data_filter)
        # print(len(data["text"]), len(data["summary"]))
        if remove_columns == []:
            data = data.map(
                fn,
                batched=True,
            )
        else:
            data = data.map(fn, batched=True, remove_columns=remove_columns)

        # data["test"] = save_test
        data.set_format("torch")
        return data

    def combine_document_field(
        self, dataset: Dataset, document_field: str = "sources"
    ) -> Dataset:
        def combine_strings(example):
            if isinstance(example[document_field], list):
                example[document_field] = " ".join(example[document_field])
            return example

        # Apply the function to the dataset
        updated_dataset = dataset.map(combine_strings)

        return updated_dataset

    def combine_columns(
        self, dataset: Dataset, document_field: str = "sources"
    ) -> Dataset:
        def combine_strings(example):
            example["summary"] = (
                f"{example['challenge']} \n {example['approach']} \n {example['outcome']}"
            )
            return example

        # Apply the function to the dataset
        updated_dataset = dataset.map(combine_strings)

        return updated_dataset

    def preprocess(
        self,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
        remove_columns: list,
    ) -> DatasetDict:
        # subclasses can implement custom behaviour by defining the preprocess fn
        return self.scrolls_preprocess(
            data=data,
            col_map=col_map,
            min_input_size=min_input_size,
            remove_columns=remove_columns,
        )
