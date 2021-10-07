import sys
from typing import List
# 打家劫舍的变型题目，应该可以用hash优化


class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        nums.sort()
        lens = len(nums)
        temp = 0
        array = []
        for i in range(lens):
            if temp > 0 and nums[i] > nums[i - 1]:
                array.append(temp)
                temp = 0
                for j in range(nums[i - 1] + 1, nums[i]):
                    array.append(0)
            temp += nums[i]
        array.append(temp)
        lens = len(array)
        if lens == 1:
            return array[0]
        prev, cur = array[0], max(array[0], array[1])
        for i in range(2, lens):
            temp = cur
            cur = max(cur, prev + array[i])
            prev = temp
        return cur


a = Solution()
print(a.deleteAndEarn([2, 2, 3, 3, 3, 4]))
