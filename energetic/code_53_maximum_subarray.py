"""
https://leetcode-cn.com/problems/maximum-subarray/
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。

"""
# !/usr/bin/python3
import sys
from typing import List


def maxSubArray(nums: List[int]) -> int:
    if len(nums) == 0:
        return 0
    ans = nums[0]
    temp = 0
    for num in nums:
        temp += num
        if temp < num:
            temp = num
        ans = max(temp, ans)

    return ans


print(maxSubArray([1]))
