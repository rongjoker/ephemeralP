import sys
from typing import List


# 桶排序中的桶大小的取值非常关键
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        maxs, mins = 0, sys.maxsize
        sets = set()
        for num in nums:
            sets.add(num)
            maxs = max(maxs, num)
            mins = min(mins, num)
        lens = len(sets)
        if lens < 2:
            return 0
        # 反证法可以证明间距不会小于space
        space = (maxs - mins) // (lens - 1)
        bucketSize = (maxs - mins) // space + 1
        lists = []
        for i in range(bucketSize):
            lists.append([])
        if maxs - mins < 2:
            return 0
        for num in sets:
            temp = lists[(num - mins) // space]
            if not temp:
                lists[(num - mins) // space].append(num)
                lists[(num - mins) // space].append(num)
            else:
                lists[(num - mins) // space][0] = min(temp[0], num)
                lists[(num - mins) // space][1] = max(temp[1], num)
        left, ans = 0, 0
        for i in range(bucketSize):
            if lists[i]:
                if lists[left]:
                    ans = max(ans,  lists[i][0] - lists[left][1])
                left = i
        return ans


a = Solution()
print(a.maximumGap([3, 6, 9, 1]))
print(a.maximumGap(
    [13684, 13701, 15157, 2344, 28728, 16001, 9900, 7367, 30607, 5408, 17186, 13230, 1598, 9766, 13083, 27618, 29065,
     9171, 2470, 20163, 5530, 20665, 14818, 4743, 24871, 27852, 8129, 4071, 17488, 30904, 1548, 16408, 1734, 17271,
     19880, 22269, 18738, 30242, 6679, 19867, 13781, 4615, 10049, 28877, 9323, 5373, 11381, 18489, 13654, 14324, 28843,
     27010, 10232, 31696, 29708, 3008, 28769, 30840, 21172, 28461, 20522, 8745, 17590, 27936, 30368, 30993, 24416,
     17472]))
print(a.maximumGap([1, 10000000]))
print(a.maximumGap(
    [15252, 16764, 27963, 7817, 26155, 20757, 3478, 22602, 20404, 6739, 16790, 10588, 16521, 6644, 20880, 15632, 27078,
     25463, 20124, 15728, 30042, 16604, 17223, 4388, 23646, 32683, 23688, 12439, 30630, 3895, 7926, 22101, 32406, 21540,
     31799, 3768, 26679, 21799, 23740]))
# print(a.maximumGap([1, 1, 1, 1, 1, 5, 5, 5, 5, 5]))
