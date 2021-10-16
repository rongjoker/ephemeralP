from typing import List


class Solution:
    sum_val = 0
    str_code = []
    ans = []

    def addOperators(self, num: str, target: int) -> List[str]:
        nums = []
        for n in num:
            nums.append(ord(n) - ord('0'))
        print(nums)
        self.sum_val += nums[0]
        self.str_code.append(str(nums[0]))
        self.backtrack(nums, 1, target, len(nums))
        return self.ans

    def backtrack(self, nums: List[int], index: int, target: int, length: int):
        if index == length:
            print('self.sum_val', self.sum_val)
            print('self.str_code', self.str_code)
            if self.sum_val == target:
                self.ans.append(''.join(self.str_code))
            return
        size = len(self.str_code)
        self.sum_val += nums[index]
        self.str_code.append('+')
        self.str_code.append(str(nums[index]))
        self.backtrack(nums, index + 1, target, length)
        # back
        del self.str_code[size:]
        self.sum_val -= nums[index]
        # track
        self.sum_val *= nums[index]
        self.str_code.append('*')
        self.str_code.append(str(nums[index]))
        self.backtrack(nums, index + 1, target, length)
        # back
        del self.str_code[size:]
        self.sum_val //= nums[index]
        # track
        self.sum_val -= nums[index]
        self.str_code.append('-')
        self.str_code.append(str(nums[index]))
        self.backtrack(nums, index + 1, target, length)
        # back
        del self.str_code[size:]
        self.sum_val += nums[index]


a = Solution()
a.addOperators('232', 8)
print(a.ans)
