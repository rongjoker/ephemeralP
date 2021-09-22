import sys
from typing import List


# quick_show_point 快慢指针系列
# 26
def remove_duplication(nums: List[int]) -> int:
    lens = len(nums)
    if lens <= 1:
        return lens
    slow = 0
    fast = 1
    while fast < lens:
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
        fast += 1
    return slow + 1


print(remove_duplication([1, 1, 1, 1, 2, 2, 3]))

# 27
