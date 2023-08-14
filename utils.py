import os
import glob
import re
import csv
import torch

def generate_binary_codes(dataset_path):
    class_dirs = glob.glob(os.path.join(dataset_path, "*"))
    binary_codes = []

    for i, class_dir in enumerate(class_dirs):
        binary_code = bin(i)[2:].zfill(8)
        files = glob.glob(os.path.join(class_dir, "*"))
        for j in range(len(files)):
            binary_codes.append(binary_code)

    return binary_codes


def get_class(dataset_root):
    class_dirs = dataset_root
    class_names = []
    sub_folders = sorted([folder for folder in os.listdir(class_dirs)
                          if os.path.isdir(os.path.join(class_dirs, folder))])
    for folder in sub_folders:
        folder_path = os.path.join(class_dirs, folder)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        for image_path in image_paths:
            filename = image_path.split("/")[-1]
            class_name = re.sub(r'\d+', '', filename)
            class_name = class_name.strip('_')
            name = class_name.split("__", 1)[0]
            class_names.append(name)

    return class_names


def get_semantic(text_path: str) -> list:
    list_sem = []
    text_dir = os.path.join(text_path)
    text_files = os.listdir(text_dir)
    for file in text_files:
        if file != '.DS_Store':
            path = text_dir + '/' + file
            kdd = os.listdir(path)
            kdd.sort()
            for text in kdd:
                file_path = path + '/' + text
                sentence = ''
                with open(file_path, 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                for word in lines:
                    sentence += word
                list_sem.append(sentence)

    return list_sem

def parse_tensor(string):
    start = string.index('[') + 1
    end = string.index(']')
    values = string[start:end].split(',')
    float_values = [float(value.strip()) for value in values]

    return float_values


def recover1(mat_path):
    data = []
    with open(mat_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.extend(row)
    data = [parse_tensor(value) if value.startswith('tensor(') else float(value) for value in data]
    total_entries = len(data)
    n = total_entries // 900  # 计算行数
    tensor_shape = (n, 900)
    tensor_data = torch.tensor(data, dtype=torch.float32).view(tensor_shape)
    print("Recover matrix shape:", tensor_data.shape)

    return tensor_data


def recover(mat_path):
    data = []
    with open(mat_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            u = []
            for i in row:
                bracket_contents = re.findall(r"\[(.*?)\]", i)
                u.append(float(bracket_contents[0]))
            data.append(u)

    data = torch.tensor(data)
    print("Recover matrix shape:", data.shape)

    return data