import sys
from typing import List


# 类似前缀和,关键在这里 ans.append(prix[q[1] + 1] ^ prix[q[0]]) ,前面要加一位,与前缀和求区间值类似


class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        prix = [0]
        a = 0
        for i in arr:
            a ^= i
            prix.append(a)
        print('p', prix)
        ans = []
        for q in queries:
            ans.append(prix[q[1] + 1] ^ prix[q[0]])
        return ans


a = Solution()
print(a.xorQueries([1, 3, 4, 8], [[0, 1], [1, 2], [0, 3], [3, 3]]))
