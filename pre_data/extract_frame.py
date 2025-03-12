import os
import random
import subprocess

def create_dataset(root_dir, output_dir, fps=4, num_videos_per_class=5):
    """
    创建数据集函数
    :param root_dir: 源数据集目录
    :param output_dir: 输出数据集目录
    :param fps: 提取帧的帧率，默认为 1fps
    :param num_videos_per_class: 每个类别随机抽取的视频数量，默认为 5
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_folders = os.listdir(root_dir)
    for class_folder in class_folders:
        class_source_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_source_path):
            print(f"跳过非目录项: {class_source_path}")
            continue

        class_output_path = os.path.join(output_dir, class_folder)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        video_files = [f for f in os.listdir(class_source_path) if f.lower().endswith(('.avi', '.mp4', '.mkv'))]
        if not video_files:
            print(f"类别 {class_folder} 中没有视频文件，跳过此类别。")
            continue

        # 每个类别随机抽取指定数量的视频
        num_videos_to_extract = min(num_videos_per_class, len(video_files))
        selected_video_files = random.sample(video_files, num_videos_to_extract)

        for video_file in selected_video_files:
            video_path = os.path.join(class_source_path, video_file)
            video_filename = video_file.split('.')[0]
            video_output_path = os.path.join(class_output_path, video_filename)
            if not os.path.exists(video_output_path):
                os.makedirs(video_output_path)

            # 检查路径权限
            if not os.access(video_output_path, os.W_OK):
                print(f"路径 {video_output_path} 无写入权限，请检查权限设置。")
                continue

            try:
                # 构建 ffmpeg 命令，增加参数提高质量
                # 将原来的 '%04d.jpg' 改为 'img_%05d.jpg'
                save_pattern = os.path.join(video_output_path, 'img_%05d.jpg')
                # -q:v 1 表示最高质量，取值范围 1-31，数值越小质量越高
                # -vf scale=-2:1080 表示将视频高度调整为 1080，宽度按比例自适应，保持画面宽高比
                # -r 指定提取帧的帧率
                cmd = f'ffmpeg -i "{video_path}" -vf "scale=-2:1080" -r {fps} -q:v 1 -vsync vfr "{save_pattern}"'
                print(f"执行命令: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0:
                    # 统计实际抽取的帧数
                    extracted_frames = len([f for f in os.listdir(video_output_path) if f.endswith('.jpg')])
                    print(f"视频 {video_file} 实际抽取了 {extracted_frames} 帧")
                else:
                    print(f"处理视频 {video_file} 时出错: {result.stderr}")
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {e}")


# 示例调用
if __name__ == "__main__":
    root_directory = r'E:\迅雷下载\UCF-101'  # 替换为实际的源数据集目录
    output_directory = r'E:\data\output'  # 替换为实际的输出数据集目录
    create_dataset(root_directory, output_directory, fps=3)


