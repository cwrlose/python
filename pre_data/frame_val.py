###生成文件 帧数 类别对比 
import os
import pandas as pd
import random


def generate_txt_file(frame_folder, labels_file, output_file):
    # 检查视频帧文件夹是否存在
    if not os.path.exists(frame_folder):
        raise FileNotFoundError(f"视频帧文件夹 {frame_folder} 不存在。")
    # 检查标签文件是否存在
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"标签文件 {labels_file} 不存在。")

    # 读取标签文件，建立标签名称到 ID 的映射
    labels_df = pd.read_csv(labels_file)
    label_name_to_id = {label: id for id, label in zip(labels_df['id'], labels_df['name'])}

    # 检查输出文件所在目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = []
    # 遍历视频帧文件夹
    for root, dirs, files in os.walk(frame_folder):
        if len(dirs) == 0:
            action_category = os.path.basename(os.path.dirname(root))
            label_id = label_name_to_id.get(action_category)
            if label_id is not None:
                frame_count = len([file for file in files if file.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
                # 构建完整路径
                full_path = root
                records.append(f"{full_path} {frame_count} {label_id}")

    if not records:
        print("没有找到匹配的图片文件进行写入，请检查文件扩展名或目录结构。")
        return

    # 打乱记录顺序
    random.shuffle(records)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(record + '\n')

    print(f"成功写入 {len(records)} 条记录到 {output_file}")
# 示例使用
frame_folder_path = r'E:\data\output'
labels_file_path = r'E:\BaiduNetdiskDownload\第12章：基于3D卷积的视频分析与动作识别\ucf_labels.csv'  # 需要替换为实际的完整路径
output_file_path = r'E:\BaiduNetdiskDownload\第12章：基于3D卷积的视频分析与动作识别\val.txt'

try:
    generate_txt_file(frame_folder_path, labels_file_path, output_file_path)
    print(f"文件已成功生成: {output_file_path}")
except Exception as e:
    print(f"发生错误: {e}")

