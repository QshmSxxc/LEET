import os
import csv
# import pickle
import dill
import json
import shutil

# def load_data_from__params_config(current_file_path):
#     current_dir = os.path.dirname(current_file_path)  # 获取当前文件夹的上一级目录
#     parent_dir = os.path.dirname(current_dir)  # 组合上一级目录和文件名，得到 params_config.json 文件的路径
#     params_config_path = os.path.join(parent_dir, 'params_config.json')  # 输出文件路径
#     params_config = load_data_from_json(params_config_path)

#     return params_config


def load_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_data_to_json(data, file_path):
    """
    写入内容到.json文件
    :param data: 字典或列表
    :param file_path:
    :return:
    """
    # 如果目录不存在，则创建目录
    create_folder(file_path)
    # 如果文件存在，先删除
    if os.path.exists(file_path):
        os.remove(file_path)
    # 打开文件（如果文件不存在会创建它），并写入数据
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # indent=4 用于格式化输出


def txt_file_add(file_directory, file_name, content_added):
    # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    # 文件路径
    file_path = os.path.join(file_directory, file_name)
    # 打开文件，并追加写入内容
    with open(file_path, 'a') as file:
        file.write(content_added)


def txt_file_write(file_directory, file_name, content_added):
    # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    # 文件路径
    file_path = os.path.join(file_directory, file_name)
    # 打开文件，并追加写入内容
    with open(file_path, 'w') as file:
        file.write(content_added)


def save_data_to_txt(data, file_path, mode='w'):
    """
    写入内容到.txt文件
    :param data: 字典或列表
    :param file_path:
    :param mode: 'w' 写入，如果文件已经存在，它会被覆盖。如果文件不存在，会创建一个新的文件; 'a' 追加，如果文件已经存在，新的内容会被写入到已有内容之后。如果文件不存在，会创建一个新的文件
    :return:
    """
    # 如果目录不存在，则创建目录
    create_folder(file_path)
    if mode == 'w':
        # 如果文件存在，先删除
        if os.path.exists(file_path):
            os.remove(file_path)
    # 打开文件（如果文件不存在会创建它），并写入数据
    with open(file_path, mode) as file:
        if isinstance(data, dict):
            file.write(str(data))
        elif isinstance(data, list):
            for item in data_list:
                file.write(item + '\n')  # 每个项后加换行符
        else:
            print("变量是其他类型")

# 保存为.csv文件
def save_data_to_csv(data_dict_list, file_path):
    # 如果目录不存在，则创建目录
    create_folder(file_path)
    # 如果文件存在，先删除
    if os.path.exists(file_path):
        os.remove(file_path)
    # 数据写入
    fields = data_dict_list[0].keys()  # 获取字典的字段名（列名）
    with open(file_path, mode="w", encoding='utf-8-sig', newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # 写入列名
        writer.writerows(data_dict_list)  # 写入数据

# 保存为.pkl文件
def save_data_to_pkl(data, file_path):
    # 如果目录不存在，则创建目录
    create_folder(file_path)
    # 如果文件存在，先删除
    if os.path.exists(file_path):
        os.remove(file_path)
    # 数据写入
    with open(file_path, 'wb') as file:
        dill.dump(data, file)

# 读取.pkl文件
def load_data_from_pkl(file_path):
    # 如果文件不存在，返回None
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在！")
    # 读取数据
    with open(file_path, 'rb') as file:
        data = dill.load(file)
    return data

# 保存图像
def save_figure_to_file(figure, file_path):
    # 如果目录不存在，则创建目录
    create_folder(file_path)
    # 如果文件存在，先删除
    if os.path.exists(file_path):
        os.remove(file_path)
    # 保存图像
    figure.savefig(file_path, dpi=300, bbox_inches='tight')  # 保存为PNG格式，DPI为300，去除多余空白

# 打开文件进行追加
def add_data_to_csv(data_dict, file_path): # data_dict是一个字典
    with open(file_path, mode='a', newline='') as file:
        # 创建一个 DictWriter 对象
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        # 如果文件大小为 0，表示新文件，则写入列标题
        if os.stat(file_path).st_size == 0:  # 文件大小为 0 表示新文件
            writer.writeheader()  # 写入列标题
        # 追加字典作为 CSV 的一行
        writer.writerow(data_dict)

# 创建文件夹
def create_folder(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        try:
            if os.path.exists(directory):
                if os.path.isfile(directory):
                    raise Exception(f"{directory} 已存在但不是文件夹，而是文件！")
            else:
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"文件夹创建失败 '{directory}': {e}")
            return

# 删除文件夹
def delete_folder(file_path):
    # 获取文件路径的目录部分
    directory = os.path.dirname(file_path)
    if directory:
        try:
            if os.path.exists(directory):
                # Use shutil.rmtree to remove non-empty directories
                shutil.rmtree(directory)
                return True
        except OSError as e:
            print(f"Failed to delete folder '{directory}': {e}")
            return False  # 如果目录删除失败，退出函数

# 删除文件
def delete_file(file_path):
    """检查文件是否存在，如果存在则删除"""
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    else:
        return False




