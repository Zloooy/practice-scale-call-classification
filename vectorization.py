from time import sleep
from itertools import islice
import os

import pandas as pd
import requests

FOLDER_ID = "b1g3t109igv0osj7e4db"
IAM_TOKEN = "t1.9euelZqSzs_LkcaTzJTHmJCaz42Oku3rnpWak8vGmorLlsmTy8mLnc_Iip3l9Pc0byBM-e96cSjI3fT3dB0eTPnvenEoyM3n9euelZqTycjHksnPmJeam8meypzKi-_8zef1656VmpmazMycl8ielY-RmpnIz5mM7_3F656VmpPJyMeSyc-Yl5qbyZ7KnMqL.yqvqfljoEebDcTal2neGFepBMQq475W6YobXmrId7Jb8L7OnZ8G5L7AraslbDlif3gGDRZGjp7V2C7j6ZWOODQ"

doc_uri = f"emb://{FOLDER_ID}/text-search-doc/latest"
query_uri = f"emb://{FOLDER_ID}/text-search-query/latest"

embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {IAM_TOKEN}",
           "x-folder-id": f"{FOLDER_ID}"}


def text_to_vector(text: str, text_type: str = "doc") -> list[float]:
    query_data = {
        "modelUri": doc_uri if text_type == "doc" else query_uri,
        "text": text if len(text) <= 7300 else text[-7300:],
    }
    result = requests.post(embed_url, json=query_data, headers=headers).json()
    print(len(text))
    print(result)
    return result["embedding"]


vectorized_texts_file = 'vectorized_files.csv'


def vectorize(transcribed_files_df):
    for i, row in enumerate(
            islice(transcribed_files_df.itertuples(index=False), 0, None)):
        if i != 0 and i % 10 == 0:
            sleep(1)
        embedding = text_to_vector(str(row.Text))
        row_df = pd.DataFrame(data=[[row.Filepath, *embedding]],
                              columns=['Filepath', *range(256)])
        row_df.to_csv(vectorized_texts_file, mode='a', header=False,
                      index=False)
        del row_df
    df = pd.read_csv(vectorized_texts_file)
    os.remove(vectorized_texts_file)
    return df
