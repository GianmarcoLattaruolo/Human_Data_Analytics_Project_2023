import glob
import logging
import os
import random
import shutil
import time
import urllib
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from hda import paths

LOGGER = logging.getLogger(__file__)


def get_audio_files(data_dir: Path) -> list[Path]:
    return [audio for audio in data_dir.iterdir() if audio.suffix == ".wav"]


def load_ESC50_meta() -> pd.DataFrame:
    df_ESC50 = pd.read_csv(paths.METADATA_DIR / "esc50.csv")
    df_ESC50["full_path"] = df_ESC50["filename"].apply(lambda x: paths.ESC_50_DIR / x)
    return df_ESC50


def load_ESC10_meta() -> pd.DataFrame:
    df_ESC50 = load_ESC50_meta()
    return df_ESC50[df_ESC50["esc10"]]


def build_target_mapping(df: pd.DataFrame) -> dict[int, str]:
    couples = df[["target", "category"]].drop_duplicates()
    return dict(couples.to_numpy())

def make_subfolders(main_dir, df):
    if "category" in df.columns:
        if len(set(df.category)) == 50:
            data_dir = os.path.join(main_dir, "Data", "ESC-50-depth")
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
        else:
            data_dir = os.path.join(main_dir, "Data", "ESC-10-depth")
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)

        for category_folder in list(set(df.category)):
            if not os.path.isdir(os.path.join(data_dir, category_folder)):
                os.mkdir(os.path.join(data_dir, category_folder))
            for old_path in df.full_path[df.category == category_folder]:
                new_path = os.path.join(
                    data_dir,
                    category_folder,
                    old_path.split("\\")[-1],
                )
                if not os.path.isfile(new_path):
                    shutil.copy(old_path, new_path)

    else:
        data_dir = os.path.join(main_dir, "Data", "ESC-US-depth")
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        for old_path in df.full_path():
            new_path = os.path.join(data_dir, "ESC-US", old_path.split("\\")[-1])
            shutil.copy(old_path, new_path)


main_dir = os.getcwd()


def reshape_US(num_files):
    def num(i):
        if i < 10:
            return "0" + str(i)
        else:
            return str(i)

    path_us = os.path.join(main_dir, "Data", "ESC-US")
    folders = [
        os.path.join(main_dir, "Data", "ESC-US", i)
        for i in os.listdir(path_us)
        if i != ".ipynb_checkpoints"
    ]
    total_files = sum([len(os.listdir(i)) for i in folders])
    folders_final = total_files // num_files + 1 * (len(folders) % num_files != 0)

    # create the new folders
    final_folders_list = []
    for i in range(folders_final):
        folder = os.path.join(main_dir, "Data", "ESC-US", num(i + 1) + "_final_bis")
        if not os.path.exists(folder):
            os.mkdir(folder)
        final_folders_list.append(folder)

    full_list_of_files = [
        [
            os.path.join(main_dir, "Data", "ESC-US", folder, i)
            for i in os.listdir(folder)
        ]
        for folder in folders
    ]  # if i[-3:]=='ogg'fa sparire alcuni files
    full_list_of_files = [i for j in full_list_of_files for i in j]

    for n, file in enumerate(full_list_of_files):
        i = n // num_files
        shutil.move(
            file,
            os.path.join(
                main_dir,
                "Data",
                "ESC-US",
                num(i + 1) + "_final_bis",
                file.split("\\")[-1],
            ),
        )
        if n % 1000 == 0:
            print(f"Moved file {n}-th of {total_files}")

    # delete the previous empty folders
    for folder in folders:
        shutil.rmtree(folder)

    # rename the new folders to cancel _final
    for i, folder in enumerate(final_folders_list):
        os.rename(folder, os.path.join(main_dir, "Data", "ESC-US", num(i + 1)))
    return


def reshape_US_leo(num_files):
    def num(i):
        if i < 10:
            return "0" + str(i)
        else:
            return str(i)

    path_us = os.path.join(main_dir, "Data", "ESC-US")
    folders = [
        os.path.join(main_dir, "Data", "ESC-US", i)
        for i in os.listdir(path_us)
        if i not in [".ipynb_checkpoints", ".DS_Store"]
    ]
    total_files = sum([len(os.listdir(i)) for i in folders])
    folders_final = total_files // num_files + 1 * (len(folders) % num_files != 0)

    # create the new folders
    final_folders_list = []
    for i in range(folders_final):
        folder = os.path.join(main_dir, "Data", "ESC-US", num(i + 1) + "_final_bis")
        if not os.path.exists(folder):
            os.mkdir(folder)
        final_folders_list.append(folder)

    full_list_of_files = [
        [
            os.path.join(main_dir, "Data", "ESC-US", folder, i)
            for i in os.listdir(folder)
        ]
        for folder in folders
    ]  # if i[-3:]=='ogg'fa sparire alcuni files
    full_list_of_files = [i for j in full_list_of_files for i in j]

    for n, file in enumerate(full_list_of_files):
        i = n // num_files

        target_dir = os.path.join(main_dir, "Data", "ESC-US", num(i + 1) + "_final_bis")
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, os.path.basename(file))
        shutil.move(file, target_path)

        if n % 1000 == 0:
            total_files = len(full_list_of_files)
            print(f"Moved file {n}-th of {total_files}")

    # delete the previous empty folders
    for folder in folders:
        shutil.rmtree(folder)

    # check if there are empty folder in os.path.join(main_dir, 'Data', 'ESC-US')
    for folder in os.listdir(os.path.join(main_dir, "Data", "ESC-US")):
        if os.path.isdir(os.path.join(main_dir, "Data", "ESC-US", folder)):
            if len(os.listdir(os.path.join(main_dir, "Data", "ESC-US", folder))) == 0:
                shutil.rmtree(os.path.join(main_dir, "Data", "ESC-US", folder))

    # rename the new folders to cancel _final
    for i, folder in enumerate(final_folders_list):
        os.rename(folder, os.path.join(main_dir, "Data", "ESC-US", num(i + 1)))


def download_dataset(name, make_subfold=False):
    if not os.path.exists(f"./data"):
        os.mkdir("Data")
    os.chdir("./data")
    """Download the dataset into current working directory.
    The labeled dataset is ESC-50, the unlabeld are ESC-US-00,ESC-US-01, ... , ESC-US-25
    but I'm not able to download them automatically from https://dataverse.harvard.edu/dataverse/karol-piczak?q=&types=files&sort=dateSort&order=desc&page=1"""

    if name == "ESC-50" and not os.path.exists(f"./{name}"):
        if not os.path.exists(f"./{name}-master.zip") and not os.path.exists(
            f"./{name}-master",
        ):
            urllib.request.urlretrieve(
                f"https://github.com/karoldvl/{name}/archive/master.zip",
                f"{name}-master.zip",
            )

        if not os.path.exists(f"./{name}-master"):
            with zipfile.ZipFile(f"{name}-master.zip", "r") as package:
                package.extractall(f"{name}-master")

        os.remove(f"{name}-master.zip")
        original = f"./{name}-master/{name}-master/audio"
        target = f"./{name}"
        shutil.move(original, target)
        original = f"./{name}-master/{name}-master/meta"
        target = f"./meta"
        shutil.move(original, target)

    if os.path.exists(f"./{name}-master"):
        shutil.rmtree(f"./{name}-master")

    else:
        print(
            "donwload it from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT&version=2.0",
        )
        pass
    os.chdir("../")

    if make_subfold:
        main_dir = os.getcwd()
        print(f"The main dir is {main_dir}")
        print("Loading the dataframes")
        df_ESC10, df_ESC50 = load_metadata(main_dir, heads=False)
        make_subfolders(main_dir, df_ESC10)
        make_subfolders(main_dir, df_ESC50)


def check_subfolder_files(path, target_extension, target_duration, sample_rate=44100):
    iter = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            iter += 1
            if file.lower().endswith(target_extension.lower()):
                file_path = os.path.join(root, file)
                audio, _ = librosa.load(file_path, sr=sample_rate)
                if len(audio) != target_duration:
                    print(
                        f"File '{file_path}' does not have the specified duration of {target_duration} ms.",
                    )
            else:
                print(
                    f"File '{file}' does not match the specified extension '{target_extension}'.",
                )
            if iter % 10000 == 0:
                print("{} files analyzed".format(iter))


def main():
    download_dataset("ESC-50")


if __name__ == "__main__":
    main()  # Call main() if this module is run, but not when imported.
