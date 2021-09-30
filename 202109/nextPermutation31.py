from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        length = len(nums)
        if length <= 1:
            return
        index = length - 2
        while index >= 0:
            if nums[index] < nums[index + 1]:
                break
            index -= 1
        if index < 0:
            nums.reverse()
        else:
            right = length - 1
            while right > index:
                if nums[right] > nums[index]:
                    break
                right -= 1
            self.swap(nums, index, right)
            self.reverse(nums, index + 1, length - 1)
        print(nums)

    def reverse(self, nums: List[int], left: int, right: int) -> None:
        while left < right:
            self.swap(nums, left, right)
            left += 1
            right -= 1

    def swap(self, nums: List[int], left: int, right: int) -> None:
        temp = nums[left]
        nums[left] = nums[right]
        nums[right] = temp


a = Solution()
a.nextPermutation([2, 3, 1, 3, 3])
# [2,3,1,3,3]
a.nextPermutation([1, 3, 2])
# [2, 1, 3]
# a.nextPermutation([3, 2, 1])
# a.nextPermutation([1, 1, 5])
