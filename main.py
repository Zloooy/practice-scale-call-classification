import sys

import pandas as pd

import cat_boost
import forest


def ap(id, status):
    with open("ap.txt", "a") as f:
        f.write(f"{id}-{str(status)}\n")


def train(dataset_path):
    df_audio_info = pd.read_csv('dataset/audio_files_info.csv')
    df_audio_info = df_audio_info.drop(columns='Sample_rate')
    df_info = pd.read_csv('info.csv', sep=';')
    df_info = df_info.drop(index=0, axis=1)
    df_info['Is incoming'] = df_info['Тип'] == 'Входящий'
    df_info['Filepath'] = 'learn_dataset/' + df_info['ID записи'].astype(
        str) + '.wav'
    df_info = df_info.drop(
        columns=['Метка', 'ID заказа звонка', 'Теги', 'Оценка',
                 'Ответственный из CRM', 'Схема', 'Куда', 'Тип', 'ID записи',
                 'Статус', 'Откуда', 'Состояние перезвона', 'Время перезвона',
                 'Запись существует', 'Длительность звонка'])
    df_info = df_info.rename(columns={
        "Длительность разговора": "Talk Duratation",
        'Кто ответил': 'Who answered',
        "Время ответа": "Answer time",
        "Новый клиент": "New client",
        "Успешный результат": "Successful",
        "Время": "Time"
    })
    df_transcribed = pd.read_csv("transcribed_files.csv")
    df = pd.merge(df_audio_info, df_transcribed, on='Filepath', how='inner')
    df = pd.merge(df, df_info, on='Filepath', how='inner')
    df = df[df['Text'].str.len() > 20]
    df_vectorized = pd.read_csv("dataset/vectorized_files.csv")
    df_ans = df[['Filepath', 'Successful']]
    df_vectorized_full = pd.merge(df_ans, df_vectorized, how='inner',
                                  on='Filepath')
    df_vectorized_full.drop(columns='Filepath', inplace=True)
    forest.train(df)
    cat_boost.train(df_vectorized_full)


def test(dataset_path):
    pass


def main():
    dataset_path = sys.argv.index("--dataset")
    if dataset_path == -1:
        print("Please provide dataset path")
        exit(1)
    dataset_path = sys.argv[dataset_path + 1]
    if "--train" in sys.argv:
        train(dataset_path)
    test(dataset_path)


if __name__ == "__main__":
    main()
