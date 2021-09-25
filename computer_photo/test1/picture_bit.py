import string
from struct import unpack
import numpy as np


# https://blog.csdn.net/rocketeerLi/article/details/84929516?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163254794216780261928295%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163254794216780261928295&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-84929516.first_rank_v2_pc_rank_v29&utm_term=python%E8%AF%BB%E5%8F%96bmp%E5%9B%BE%E7%89%87

class Picture_read:
    def read_file(self, path: string):
        f = open(path, 'rb')
        # 文件头 14 bit
        self.by_type = unpack("<h", f.read(2))[0]
        self.by_size = unpack("<i", f.read(4))[0]
        print('by_type:', self.by_type)
        print('by_size:', self.by_size)
        f.read(4)
        # print('rr:', unpack("<h", f.read(2))[0])
        # print('rr:', unpack("<h", f.read(2))[0])
        offset = unpack("<i", f.read(4))[0]
        print('offset:', offset)
        #
        #
        # 位图信息头 40bit
        self.bi_size = unpack("<i", f.read(4))[0]
        self.width = unpack("<i", f.read(4))[0]
        self.hight = unpack("<i", f.read(4))[0]
        self.ps = self.width * self.hight
        print('width', self.width)
        print('hight', self.hight)
        print('ps', self.ps)
        # print('rr:', unpack("<h", f.read(2))[0])
        # print('rr:', unpack("<h", f.read(2))[0])

        # bit_compression = unpack("<i", f.read(4))[0]  # 压缩类型
        # bit_size_img = unpack("<i", f.read(4))[0]  # 图像大小
        # bp_x = unpack("<i", f.read(4))[0]  # 水平分辨率
        # bp_y = unpack("<i", f.read(4))[0]  # 垂直分辨率
        # f.read(4)
        # f.read(4)
        f.read(28)
        self.bmp_bit = []
        # f.seek(offset)
        for i in range(self.hight):
            bmp_data_row = []
            count = 0
            for column in range(self.width):
                bmp_data_row.append(
                    [unpack("<B", f.read(1))[0], unpack("<B", f.read(1))[0], unpack("<B", f.read(1))[0]])
                count += 3
            while count % 4 != 0:
                f.read(1)
                count += 1
            self.bmp_bit.append(bmp_data_row)
        print(self.bmp_bit)
        # for row in range(100):
        #     for column in range(self.width):
        #     print('rr:', unpack("<h", f.read(2))[0])

        f.close()


a = Picture_read()
a.read_file("/Users/zhangshipeng/Downloads/fd/compute_photo/test1/demo.bmp")
# pip3 install numpy
