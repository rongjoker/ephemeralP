from typing import List


# 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        length = len(nums)
        if length == 1:
            return nums[0]

        min_val, max_val, ans = nums[0], nums[0], nums[0]
        for i in range(1, length):
            temp = min(min(min_val * nums[i], nums[i]), max_val * nums[i])
            max_val = max(max(min_val * nums[i], nums[i]), max_val * nums[i])
            min_val = temp
            ans = max(ans, max_val)
        return ans


a = Solution()
print(a.maxProduct([-2, 0, -1]))
print(a.maxProduct([2, 3, -2, 4]))
# [-2,0,-1] 0
# [2,3,-2,4] 6
