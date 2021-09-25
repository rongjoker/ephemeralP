import string
from struct import unpack
import numpy as np


# https://blog.csdn.net/rocketeerLi/article/details/84929516?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163254794216780261928295%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163254794216780261928295&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-84929516.first_rank_v2_pc_rank_v29&utm_term=python%E8%AF%BB%E5%8F%96bmp%E5%9B%BE%E7%89%87

class Picture_read:
    def read_file(self, path: string):
        f = open(path, 'rb')
        # 文件头 14 bit
        by_type = unpack("<h", f.read(2))[0]
        by_size = unpack("<i", f.read(4))[0]
        f.read(4)
        offset = unpack("<i", f.read(4))[0]
        # 位图信息头 40bit
        bi_size = unpack("<i", f.read(4))[0]
        width = unpack("<i", f.read(4))[0]
        height = unpack("<i", f.read(4))[0]
        ps = width * height
        f.read(28)
        # 文件bit数
        bit_num = unpack('<h', f.read(2))[0]
        f.seek(50)
        cn = unpack("<i", f.read(4))[0]
        f.seek(54)
        bmp_bit = []
        for i in range(height):
            bmp_data_row = []
            for column in range(width):
                bmp_data_row.append(
                    [unpack("B", f.read(1))[0], unpack("B", f.read(1))[0], unpack("B", f.read(1))[0]])
            f.seek(1)
            bmp_bit.append(bmp_data_row)
        #
        f.seek(offset)
        px = np.zeros((height, width, 3), dtype=np.int32)

        f.close()


a = Picture_read()
a.read_file("/Users/zhangshipeng/Downloads/fd/compute_photo/test1/demo.bmp")
# pip3 install numpy
