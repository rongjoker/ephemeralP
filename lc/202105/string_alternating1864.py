import sys
from typing import List
# 一个简单的思维题目 1864. 构成交替字符串需要的最小交换次数
# https://leetcode-cn.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/


class Solution:
    def minSwaps(self, s: str) -> int:
        lens = len(s)
        len0, len1 = 0, 0
        for i in range(lens):
            if s[i] == '0':
                len0 += 1
            else:
                len1 += 1
        if len0 - len1 > 1 or len0 - len1 < -1:
            return -1
        # 0101010
        if len0 > len1:
            len0, len1 = 0, 0
            for i in range(lens):
                if i % 2 == 0 and s[i] == '1':
                    len0 += 1
                elif i % 2 == 1 and s[i] == '0':
                    len1 += 1
            return max(len0, len1)
        # 1010101
        elif len0 < len1:
            len0, len1 = 0, 0
            for i in range(lens):
                if i % 2 == 0 and s[i] == '0':
                    len1 += 1
                elif i % 2 == 1 and s[i] == '1':
                    len0 += 1
            return max(len0, len1)
        else:
            len0, len1 = 0, 0
            for i in range(lens):
                if i % 2 == 0 and s[i] == '1':
                    len0 += 1
                elif i % 2 == 1 and s[i] == '0':
                    len1 += 1
            ans = max(len0, len1)
            len0, len1 = 0, 0
            for i in range(lens):
                if i % 2 == 0 and s[i] == '0':
                    len1 += 1
                elif i % 2 == 1 and s[i] == '1':
                    len0 += 1
            return min(ans, max(len0, len1))


a = Solution()
print(a.minSwaps(
    "01"))
