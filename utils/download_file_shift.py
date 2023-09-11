import os
import shutil

# 当前目录
dir_paths = [
             "C:\\Users\\rongjoker\AppData\Local\Temp\Highlights/PLAYERUNKNOWN'S BATTLEGROUNDS",
             "C:\\Users\\rongjoker\Videos\pubg"]

# 获取当前目录下的所有文件
# files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
#
# # 遍历文件列表，输出文件名
# for file in files:
#     print(file)
#     print(os.stat(file).st_size//(1024*1024))
#     shutil.move("E:\\pt.txt", "D:\\pythonOU\\cryt")
for dir_path in dir_paths:
    for file in os.listdir(dir_path):
        print('dir_path:', dir_path)
        print('file:', file)
        print(os.stat(os.path.join(dir_path, file)).st_size // (1024 * 1024))
        shutil.move(os.path.join(dir_path, file), "E:\\video\\nvidia\\pubg")
