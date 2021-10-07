# coding=utf-8
import sys


# 节约空间，不过容易写错
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0':
            return 0
        lenx = len(s)
        prev = 1
        current = 1
        for i in range(1, lenx):
            if s[i] == '0':
                if s[i - 1] == '0' or s[i - 1] > '2':
                    return 0
                else:
                    temp = current
                    current = prev
                    prev = temp
            elif s[i - 1] == '0' or s[i - 1:i + 1] > '26':
                prev = current
            else:
                temp = current
                current = prev + current
                prev = temp
        return current


a = Solution()
print(a.numDecodings("10"))
