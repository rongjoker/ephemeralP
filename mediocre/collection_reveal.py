import sys
import os
import time
import math

# 创建集合的推导式语法(推导式也可以用于推导集合)
set4 = {num for num in range(1, 100) if num % 3 == 0 or num % 5 == 0}
print(set4)

# Python中的集合跟数学上的集合是一致的，不允许有重复元素，而且可以进行交集、并集、差集等运算

set1 = {1, 2, 3, 3, 3, 2}
print(set1)
print('Length =', len(set1))


def main():
    content = '北京欢迎你为你开天辟地…………'
    while True:
        # 清理屏幕上的输出
        os.system('clear')  # os.system('clear')
        print(content)
        # 休眠200毫秒
        time.sleep(0.2)
        content = content[1:] + content[0]


# if __name__ == '__main__':
#     main()

def mathematics(num=123):
    return math.acos(num)


# print(math.pow(1/2, 1/2))
print(mathematics(1))


items1 = dict(one=1, two=2, three=3, four=4)

print(items1['one'])

for k in items1:
    print(f'{k}:{items1[k]}')



