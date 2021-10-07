from typing import List


class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        len1, len2 = len(nums1), len(nums2)
        # 这种定义二维数组的方法有问题
        # dp = [[0] * (len2 + 1)] * (len1 + 1)
        dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        ans = 0
        for i in range(len1):
            for j in range(len2):
                if nums1[i] == nums2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    ans = max(ans, dp[i + 1][j + 1])
        return ans


a = Solution()
print(a.findLength([1, 0, 0, 0],
                   [1, 0, 0, 1, 1]))
