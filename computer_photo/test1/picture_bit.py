import string
from struct import unpack
import numpy as np


class PictureRead:
    def read_file(self, path: string):
        f = open(path, 'rb')
        # 文件头 14 bit
        by_type = unpack("<h", f.read(2))[0]
        by_size = unpack("<i", f.read(4))[0]
        f.seek(10)
        offset = unpack("<i", f.read(4))[0]
        # 位图信息头 40bit
        f.seek(f.tell() + 4)
        width = unpack("<i", f.read(4))[0]
        height = unpack("<i", f.read(4))[0]
        ps = width * height
        f.seek(28)
        # 文件bit数
        bit_num = unpack('<h', f.read(2))[0]
        f.seek(50)
        cn = unpack("<i", f.read(4))[0]
        f.seek(54)
        color_table_num = 2 ** int(bit_num)
        color_table = np.zeros((color_table_num, 3), dtype=np.int32)
        for i in range(color_table_num):
            color_table[i][0] = unpack('B', f.read(1))[0]
            color_table[i][1] = unpack('B', f.read(1))[0]
            color_table[i][2] = unpack('B', f.read(1))[0]
            f.read(1)
        f.seek(offset)
        px = np.zeros((height, width, 3), dtype=np.int32)
        for i in range(height):
            for j in range(width):
                index = unpack('B', f.read(1))[0]
                px[height - i - 1, j] = color_table[index]
        f.close()
        print('宽度:', width)
        print('高度:', height)
        print('像素:', ps)
        print('文件大小:', format(by_size * 8 / 1024 / 1024, '.2f'), 'Mb')
        print('区间灰度值:')
        print(px[0:200, 100:200, 0])


a = PictureRead()
a.read_file("/Users/zhangshipeng/Downloads/fd/compute_photo/test1/demo.bmp")
