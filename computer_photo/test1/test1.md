```python
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
        colors_num = 2 ** int(bit_num)
        colors = np.zeros((colors_num, 3), dtype=np.int32)
        for i in range(colors_num):
            colors[i][0] = unpack('B', f.read(1))[0]
            colors[i][1] = unpack('B', f.read(1))[0]
            colors[i][2] = unpack('B', f.read(1))[0]
            f.read(1)
        f.seek(offset)
        px = np.zeros((height, width, 3), dtype=np.int32)
        for i in range(height):
            for j in range(width):
                index = unpack('B', f.read(1))[0]
                px[height - i - 1, j] = colors[index]
        f.close()
        print('宽度:', width)
        print('高度:', height)
        print('像素:', ps)
        print('文件大小:', format(by_size * 8 / 1024 / 1024, '.2f'), 'Mb')
        print('区间灰度值:')
        print(px[0:200, 100:200, 0])

        
a = PictureRead()
a.read_file("demo.bmp")
```

输出：
```

宽度: 2000
高度: 1312
像素: 2624000
文件大小: 20.03 Mb
区间灰度值:
[[42 42 43 ... 44 46 43]
 [40 41 42 ... 46 47 46]
 [42 42 42 ... 45 45 45]
 ...
 [53 53 52 ... 54 54 54]
 [52 52 51 ... 52 54 53]
 [52 53 52 ... 54 54 54]]


```