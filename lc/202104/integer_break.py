# 343. 整数拆分

import sys


class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in (range(3, n+1)):
            for j in (range(1, i)):
                dp[i] = max(dp[i], max(j * dp[i - j], j*(i-j)))
        return dp[n]


a = Solution()
print(a.integerBreak(57))



