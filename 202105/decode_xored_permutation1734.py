import sys
from typing import List
# 利用间隔求出第一个值，然后转化成简单难度的题目


class Solution:
    def decode(self, encoded: List[int]) -> List[int]:
        lens = len(encoded)
        aka = 0
        for i in range(1, lens + 2):
            aka ^= i
        b = 0
        for i in range(1, lens, 2):
            b ^= encoded[i]
        first = aka ^ b
        ans = [first]
        for e in encoded:
            first = first ^ e
            ans.append(first)
        return ans


a = Solution()
print(a.decode([3, 1]))
