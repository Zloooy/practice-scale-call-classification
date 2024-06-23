import sys

import pandas as pd


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
    df.to_csv('train.csv', index=False)
    forest.train(df)


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
