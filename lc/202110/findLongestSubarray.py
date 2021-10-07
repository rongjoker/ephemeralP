
# find findLongestSubarray
from typing import List


def findLongestSubarray(array: List[str]) -> List[str]:
    length = len(array)
    if length <= 1:
        return []
    prefix = [0] * (length + 1)
    for index, s in enumerate(array):
        temp = 1 if ord(s[0]) > ord('9') else -1
        prefix[index + 1] = prefix[index] + temp
    left, right = 0, 0
    for index in range(length + 1):
        for j in range(length, -1, -1):
            if prefix[j] - prefix[index] == 0:
                if j - index > right - left:
                    left = index
                    right = j
                    break
    return array[left:right]


# 前缀和 + dict
def findLongestSubarray2(array: List[str]) -> List[str]:
    length = len(array)
    if length <= 1:
        return []
    prefix = [0] * (length + 1)
    dict = {0: 0}
    left, right = 0, 0
    for index, s in enumerate(array):
        temp = 1 if ord(s[0]) > ord('9') else -1
        pref_sum = prefix[index] + temp
        prefix[index + 1] = pref_sum
        if pref_sum in dict:
            prev = dict[pref_sum]
            if index + 1 - prev > right - left:
                left = prev
                right = index + 1
        else:
            dict[pref_sum] = index + 1
    return array[left:right]


print('---------  test 3 arguments -------')
print(findLongestSubarray(['A', 'B',  '1', 'B', '2']))
print(findLongestSubarray2(['A', 'B',  '1', 'B', '2']))
print(findLongestSubarray(["A", "A"]))
print(findLongestSubarray2(["A", "A"]))
