import sys
from threading import Thread
import time
import os

import requests


def read():
    # 一次性读取整个文件内容
    with open('/Users/rongjoker/python_world/传参.txt', 'r', encoding='utf-8') as f:
        print(f.read())


def read_line():
    # 通过for-in循环逐行读取
    with open('/Users/rongjoker/python_world/传参.txt', mode='r') as f:
        for line in f:
            print(line, end='')
            time.sleep(0.5)
    print()


def read_to():
    # 读取文件按行读取到列表中
    with open('传参.txt') as f:
        lines = f.readlines()
    print(lines)


# read_line()

def rename_file(path='/Users/rongjoker/python_world/'):
    try:
        for file in os.listdir(path):
            print(f'file:{file}')
            print(os.path.abspath(file))
            if file.endswith('jpg'):
                print(f'i am jpg:{file}')
                # os.renames('/Users/rongjoker/python_world/'+file, '/Users/rongjoker/python_world/'+f'joker.jpg')
    except Exception as e:
        print(e)


def run_x(url):
    filename = url[url.rfind('/') + 1:]
    resp = requests.get(url)
    print(f'filename:{filename};url:{url}')
    with open('/Users/rongjoker/python_world/' + filename, 'wb') as f:
        f.write(resp.content)


run_x('https://pic1.zhimg.com/80/v2-3cb9eeb68ffbaa3e14267a7404fa5585_1440w.jpg')
rename_file()



