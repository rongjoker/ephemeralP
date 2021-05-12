import sys
from typing import List


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        self.quickSort(nums, 0, len(nums) - 1)

    def quickSort(self, nums: List[int], le: int, r: int) -> None:
        # 这一步不可少
        if le >= r:
            return
        pivot = nums[le]
        left, right = le, r
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            if left < right:
                nums[left] = nums[right]
                left += 1
            while left < right and nums[left] <= pivot:
                left += 1
            if left < right:
                nums[right] = nums[left]
                right -= 1
        nums[left] = pivot
        self.quickSort(nums, le, left - 1)
        self.quickSort(nums, left + 1, r)


a = Solution()
a.sortColors([2, 0, 2, 1, 1, 0])
# a.sortColors([1, 3, 2])
