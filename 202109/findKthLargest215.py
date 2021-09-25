from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quick_sort(nums, k - 1, 0, len(nums) - 1)

    def quick_sort(self, nums: List[int], k: int, left: int, right: int) -> int:
        l, r = left, right
        pivot = nums[l]
        while l < r:
            while l < r and pivot >= nums[r]:
                r -= 1
            if l < r:
                nums[l] = nums[r]
                l += 1
            while l < r and nums[l] >= pivot:
                l += 1
            if l < r:
                nums[r] = nums[l]
                r -= 1
        nums[l] = pivot
        if l == k:
            print(nums)
            return nums[l]
        elif l < k:
            return self.quick_sort(nums, k, l + 1, right)
        else:
            return self.quick_sort(nums, k, left, l - 1)


a = Solution()
print(a.findKthLargest([3, 2, 1, 5, 6, 4], 2))
