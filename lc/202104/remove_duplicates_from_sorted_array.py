import sys
from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        lens = len(nums)
        if lens == 1:
            return 1
        left = 0
        for right in range(1, lens):
            if nums[right] != nums[left]:
                left = left + 1
                nums[left] = nums[right]
        return left + 1


a = Solution()
print(a.removeDuplicates([1, 1, 2]))
