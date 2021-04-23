import sys
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        lists = []
        lens = len(nums)
        if lens < 3:
            return lists
        nums.sort()
        if nums[0] + nums[1] + nums[2] > 0:
            return lists
        if nums[lens - 1] + nums[lens - 2] + nums[lens - 3] < 0:
            return lists
        for i in range(0, lens):
            # cut
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = lens - 1
            while k > j:
                if nums[i] + nums[j] + nums[k] == 0:
                    lists.append([nums[i], nums[j], nums[k]])
                    while k > j and nums[k] == nums[k - 1]:
                        k -= 1
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    k -= 1
                    j += 1
                elif nums[i] + nums[j] + nums[k] < 0:
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    j += 1
                else:
                    while k > j and nums[k] == nums[k - 1]:
                        k -= 1
                    k -= 1
        return lists


a = Solution()
print(a.threeSum([-1, 0, 1, 2, -1, -4]))
print(100 & 1)
