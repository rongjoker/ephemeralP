import sys
from typing import List
# 18. 四数之和 https://leetcode-cn.com/problems/4sum/
# a d 两个指针循环，一个从左，一个从右
# b c 两个指针类似三数之和里的b c逼近迭代


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        lists = []
        lens = len(nums)
        if lens < 4:
            return lists
        nums.sort()
        for i in range(0, lens - 3):
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = lens - 1
            while j > 2:
                if j < lens - 1 and nums[j] == nums[j + 1]:
                    j = j - 1
                    continue
                left = i + 1
                right = j - 1
                while left < right:
                    sums = nums[i] + nums[j] + nums[left] + nums[right]
                    if sums == target:
                        lists.append([nums[i], nums[j], nums[left], nums[right]])
                        while nums[left] == nums[left + 1] and left < right:
                            left = left + 1
                        left = left + 1
                        while nums[right] == nums[right - 1] and left < right:
                            right = right - 1
                        right = right - 1
                    elif sums < target:
                        left = left + 1
                    else:
                        right = right - 1
                j = j - 1
        return lists


a = Solution()
print(a.fourSum([-2, -1, -1, 1, 1, 2, 2], 0))
