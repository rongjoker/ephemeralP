from typing import List

import numpy as np
import matplotlib.pyplot as mp
import operator as opr
from functools import reduce

# 三维数组，nums[0][0] = [10,10,10]
x = np.zeros((2, 2, 3), dtype=np.int32)
x[0, 0] = 10
print(x)

# graph
# mp.figure(1)
# mp.figure(2)
# mp.show()

print(opr.eq('joker', 'hello'))
print(opr.ge(10, 10))
print(opr.ge(10, 8))


def sort_quick(nums: List[int], left: int, right: int):
    if left >= right:
        return
    temp_l, temp_r = left, right
    pivot = nums[temp_l]
    while temp_l < temp_r:
        while nums[temp_r] > pivot and temp_l < temp_r:
            temp_r -= 1
        if temp_l < temp_r:
            nums[temp_l] = nums[temp_r]
            temp_l += 1
        while nums[temp_l] < pivot and temp_l < temp_r:
            temp_l += 1
        if temp_l < temp_r:
            nums[temp_r] = nums[temp_l]
            temp_r -= 1
    nums[temp_l] = pivot
    sort_quick(nums, left, temp_l - 1)
    sort_quick(nums, temp_l + 1, right)


def simple2():
    pass


def show_key(nums: List[int], left=0, right=-1):
    print(nums[left])
    print(nums[right])


ss = [1, 7, 6, 4, 2]
print(ss)
print(opr.and_(1, 2))
print(opr.xor(1, 2))
print(opr.or_(1, 2))
print('-----test list----')
list1 = [1, 2, 3]
list2 = [7, 9, 8, 3, 2, 11]
list1.extend(list2)
print(list1)
print(list1.count(3))
print(list1.index(3))
print(list1[-1])
list1.pop(list1.index(9))
print(list1)
list1.pop()
sort_quick(list1, 0, len(list1) - 1)
print(list1)
list1.reverse()
print(list1)
list1.remove(1)
print(list1)
list1.sort()
print(list1)
list1.sort(reverse=True)
print(list1)
print('-----test dict----')
dictionary = {'joker': 1, 'jj': 2}
print('kkk:', dictionary.keys())
for i in dictionary.keys():
    print('i:', i)
dictionary.popitem()
print('joker' in dictionary)
print('joker1' not in dictionary)
print(list1[1:2])
print('-----test string----')
code = ' I am Programmer '
print(code[-1])
print(code[3:-1])
print(code.lower())
print(code)
code.strip()
print(code)
code.lstrip()
print(code)
code2 = code.lstrip()
print(code2)
code3 = code.upper()
print(code3)
show_key(list1, 2)
# ord : char
print(ord('z') - ord('a'))
# enumerate
code_list = ['joker', 'jj', 'rong', 'type']
for i, k in enumerate(code_list):
    print(i, ':', k)
# math
print(pow(10, 3))
# range
print(list(range(8)))
# lambda reduce
sum2 = reduce(lambda x1, y1: x1 + y1, [1, 2, 3, 4, 5, 6])
print('sum2:', sum2)
# iter and next
dictionary2 = {'joker': 1, 'jj': 2, 'mm': 3, 'aa': 4}
it = iter(dictionary2)
while True:
    try:
        x = next(it)
        print(x, ':', dictionary2[x])
    except StopIteration:
        print('stop!')
        break
# 哈希表 + items + next 判断
citiesA1 = {path for path in dictionary2}
citiesA2 = {4}
print(citiesA1)
print('word:', next(k for k, v in dictionary2.items() if v in citiesA2))
print('------- test next over -----')


def destCity(self, paths: List[List[str]]) -> str:
    hashmap = {}
    for path in paths:
        hashmap[path[0]] = path[1]
    p, ans = paths[0][0], hashmap[paths[0][0]]
    while p in hashmap:
        ans = hashmap[p]
        p = ans
    return ans


def destCity2(self, paths: List[List[str]]) -> str:
    citiesA = {path[0] for path in paths}
    return next(path[1] for path in paths if path[1] not in citiesA)
