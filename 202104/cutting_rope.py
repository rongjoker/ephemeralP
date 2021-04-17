import sys


class Solution:
    def cuttingRope(self, n: int) -> int:
        lists = [0] * (n + 1)
        lists[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i):
                lists[i] = max(lists[i], max((j * (i - j)) % 1000000007, ( lists[i - j]) % 1000000007))
            print(lists[i])
        return lists[n]


a = Solution()
a.cuttingRope(100)
