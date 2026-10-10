import os
import json
from pathlib import Path


def load_config_from_json():
    # current_file_path = Path(__file__).resolve()  # 获取当前脚本所在的文件夹路径
    current_dir = Path(__file__).resolve().parent  # 获取当前文件夹的上一级目录
    json_file_path = os.path.join(current_dir, 'params_config.json')

    with open(json_file_path, "r") as file:
        data = json.load(file)

    # 拼接数据和图像路径
    project_dir = current_dir.parent.parent
    data["data_figure_dir"] = project_dir / data["data_figure_dir"].lstrip("\\/")
    return data

if __name__ == "__main__":
    print(load_config_from_json())
