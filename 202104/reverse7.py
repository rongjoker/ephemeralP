import sys


class Solution:
    def reverse(self, x: int) -> int:
        flag = False
        if x < 0:
            flag = True
            x = -x
        ans = 0
        while x > 0:
            ans = ans * 10
            ans += x % 10
            x /= 10
        return - ans if flag else ans


a = Solution()
print(a.reverse(123))
