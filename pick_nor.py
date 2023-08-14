import os
import random
import shutil

def create_random_dataset(cub_path, output_path):
    # 获取 CUB 数据集中的类别列表
    class_list = sorted(os.listdir(cub_path))
    random.shuffle(class_list)

    for class_name in class_list:
        class_dir = os.path.join(cub_path, class_name)
        if os.path.isdir(class_dir):
            # 找到当前类别的图像文件列表
            image_files = os.listdir(class_dir)
            random.shuffle(image_files)

            # 从当前类别中选择一个图像
            image_file = image_files[0]

            # 源文件路径
            src_path = os.path.join(class_dir, image_file)

            # 目标文件路径
            dest_path = os.path.join(output_path, class_name)

            # 创建目标文件夹（如果不存在）
            os.makedirs(dest_path, exist_ok=True)

            # 复制图像到目标文件夹
            shutil.copy(src_path, dest_path)

            print(f"Copied {image_file} from {class_name} to {dest_path}")

    print("Random dataset creation complete.")

# 设置 CUB 数据集路径和输出路径
cub_dataset_path = "/Users/jmy/Desktop/Vaegan_gpu_0722/Mini_CUB1/images"
output_dataset_path = "/Users/jmy/Desktop/Vaegan_gpu_0722/test/set4"
#cub_dataset_path = "/Users/chongzhang/PycharmProjects/Vaegan_gpu/Train_CUB_2/images"
#output_dataset_path = "/Users/chongzhang/PycharmProjects/Vaegan_gpu/test/set2"

# 创建随机数据集
create_random_dataset(cub_dataset_path, output_dataset_path)