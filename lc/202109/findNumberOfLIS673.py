from typing import List


class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        dp = [0] * 100
        print(dp)
        return dp[0]


a = Solution()
a.findNumberOfLIS([1, 2, 3, 4, 4, 4])
