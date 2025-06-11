#download a link into ../data/
import requests
import pandas as pd
from pathlib import Path

def download_data():
    urls = {'train':"https://huggingface.co/datasets/kensho/DocFinQA/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            'test':"https://huggingface.co/datasets/kensho/DocFinQA/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
            'val':"https://huggingface.co/datasets/kensho/DocFinQA/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"}

    #download the files into ../data/

    for split, url in urls.items():
        response = requests.get(url)
        with open(f"../data/{split}.parquet", "wb") as file:
            file.write(response.content)

if __name__ == "__main__":
    download_data()