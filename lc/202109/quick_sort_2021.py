from typing import List


def quick_sort(nums: List[int], left: int, right: int):
    if left >= right:
        return
    l, r = left, right
    temp = nums[l]
    while l < r:
        while l < r and nums[r] >= temp:
            r -= 1
        if l < r:
            nums[l] = nums[r]
            l += 1
        while l < r and nums[l] <= temp:
            l += 1
        if l < r:
            nums[r] = nums[l]
            r -= 1
    nums[r] = temp
    quick_sort(nums, left, l - 1)
    quick_sort(nums, l + 1, right)


ss = [1, 7, 9, 10, 2, 4, 3]
quick_sort(ss, 0, len(ss) - 1)
print(ss)
