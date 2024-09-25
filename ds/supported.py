import logging
from typing import Mapping, Sequence, Union

from ds.arxiv import Arxiv
from ds.bigpatent import BigPatent
from ds.govreport import GovReport
from ds.hfdataset import HFDataset
from ds.pubmed import Pubmed
from ds.wispermed import LaySum

translate_dataset_name = {
    "arxiv": Arxiv,
    "govreport": GovReport,
    "bigpatent": BigPatent,
    "pubmed": Pubmed,
    "wispermed": LaySum,
}


def load_dataset(
    dataset: str,
    preview: bool = False,
    samples: Union[int, str] = "max",
    min_input_size: int = 0,
    load_csv: bool = False,
    data_files: Union[
        dict, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None
    ] = None,
) -> HFDataset:
    logging.info(f"Preparing dataset {dataset}")

    assert dataset in translate_dataset_name

    return translate_dataset_name[dataset](
        preview=preview,
        samples=samples,
        min_input_size=min_input_size,
        #load_csv=load_csv,
        #data_files=data_files,
    )
