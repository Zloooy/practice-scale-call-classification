import datetime
import os
import sys

import numpy as np
import pandas as pd
import librosa
import whisperx

import cat_boost
import forest
import vectorization
import vectorized_forest


def ap(id, status):
    with open("ap.txt", "a") as f:
        f.write(f"{id}-{str(status)}\n")


def train(dataset_path):
    df_audio_info = pd.read_csv('dataset/audio_files_info.csv')
    df_audio_info = df_audio_info.drop(columns='Sample_rate')
    df_info = pd.read_csv('dataset/info.csv', sep=';')
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
    df_transcribed = pd.read_csv("dataset/transcribed_files.csv")
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
    vectorized_forest.train(df_vectorized_full)


def convert_timestamp_to_datetime(timestamp):
    try:
        return datetime.datetime.fromtimestamp(float(timestamp)).strftime(
            '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None


def generate_audio_files_info(data_dir):
    def read_audio_file(file_path):
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            timestamp = os.path.splitext(os.path.basename(file_path))[
                0]
            return True, timestamp, audio_data, sample_rate
        except Exception:
            ap(file_path, "Fail")
            return False, os.path.splitext(os.path.basename(file_path))[
                0], None, None

    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    audio_info_list = []

    for wav_file in wav_files:
        wav_path = os.path.join(data_dir, wav_file)
        is_supported, timestamp, audio, sample_rate = read_audio_file(wav_path)

        if is_supported:
            duration_sec = librosa.get_duration(y=audio, sr=sample_rate)
            num_samples = len(audio)
        else:
            duration_sec = None
            num_samples = None

        audio_info = {
            'Timestamp': timestamp,
            'Filepath': wav_path,
            'Duration_sec': duration_sec,
            'Sample_rate': sample_rate,
            'Num_samples': num_samples,
            'Is_supported': is_supported
        }
        audio_info_list.append(audio_info)

    df_audio = pd.DataFrame(audio_info_list)
    df_audio['Timestamp'] = df_audio['Timestamp'].apply(
        convert_timestamp_to_datetime)
    df_audio = df_audio[df_audio['Is_supported']]
    return df_audio


def generate_transcribed_files(df_audio):
    transcribed_files_table = "transcribed_files.csv"
    device = "cuda"
    compute_type = "float16"
    from itertools import groupby
    from operator import itemgetter
    model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
        language="ru",
        asr_options={"without_timestamps": True},
    )
    model_a, metadata = whisperx.load_align_model(language_code="ru",
                                                  device=device)
    skip_rows = 0
    if not os.path.exists(transcribed_files_table):
        header_df = pd.DataFrame({"Filepath": [], "Text": []})
        header_df.to_csv(transcribed_files_table, mode="w", header=True,
                         index=False)
        del header_df
    else:
        skip_rows = pd.read_csv(transcribed_files_table).shape[0] - 1
    print("Длина df_audio: ", len(df_audio))
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token="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",
        device=device,
    )
    for audio_file in df_audio.query("Is_supported").tail(-skip_rows)[
        "Filepath"]:
        batch_size = 16

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        print(result["segments"])  # before alignment
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        print(result["segments"])  # after alignment

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        print(result["segments"])  # segments are now assigned speaker IDs

        def safe_itemgetter(key, default=None):
            return lambda item: item.get(key, default)

        row_df = pd.DataFrame(
            [
                {
                    "Filepath": audio_file,
                    "Text": "\n".join(
                        map(
                            lambda replics: " ".join(
                                map(safe_itemgetter("text", ""), replics)
                            ),
                            map(
                                itemgetter(1),
                                groupby(result["segments"],
                                        key=safe_itemgetter("speaker")),
                            ),
                        )
                    ),
                }
            ]
        )
        row_df.to_csv(transcribed_files_table, mode="a", header=False,
                      index=False)
        del row_df
    df = pd.read_csv(transcribed_files_table)
    os.remove(transcribed_files_table)
    return df


def test(dataset_path):
    df_info = pd.read_csv(f'{dataset_path}/info.csv', sep=';')
    df_info['Is incoming'] = df_info['Тип'] == 'Входящий'
    df_info['Filepath'] = f'{dataset_path}/learn_dataset/' + df_info[
        'ID записи'].astype(
        str) + '.wav'
    df_info = df_info.drop(
        columns=['Метка', 'ID заказа звонка', 'Теги', 'Оценка',
                 'Ответственный из CRM', 'Схема', 'Куда', 'Тип',
                 'Статус', 'Откуда', 'Состояние перезвона', 'Время перезвона',
                 'Запись существует', 'Длительность звонка'])
    df_info = df_info.rename(columns={
        "Длительность разговора": "Talk Duratation",
        'Кто ответил': 'Who answered',
        "Время ответа": "Answer time",
        "Новый клиент": "New client",
        "Время": "Time"
    })
    df_audio_info = generate_audio_files_info(
        f'{dataset_path}/learn_dataset')

    df_transcribed = generate_transcribed_files(df_audio_info)
    df = pd.merge(df_audio_info, df_transcribed, on='Filepath', how='inner')
    df = pd.merge(df, df_info, on='Filepath', how='inner')
    df = df[df['Text'].str.len() > 20]
    req = df['ID записи']
    df = df.drop(columns=['ID записи'])
    df_vectorized = vectorization.vectorize(df)
    res = np.mean(
        [forest.predict(df) * 2, cat_boost.predict(df_vectorized) * 1.5,
         vectorized_forest.predict(df_vectorized) * 1])
    for i in range(len(req)):
        t = res[i] > 1.5
        ap(req[i], t)


def main():
    dataset_path = sys.argv.index("--dataset")
    if dataset_path == -1:
        print("Please provide dataset path")
        exit(1)
    dataset_path = sys.argv[dataset_path + 1]
    if "--learn" in sys.argv:
        train(dataset_path)
    test(dataset_path)


if __name__ == "__main__":
    main()
