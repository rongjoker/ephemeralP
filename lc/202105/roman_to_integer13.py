import sys
from typing import List


class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        lens = len(s)
        ans = 0
        for i in range(lens):
            if i < lens - 1 and dic[s[i]] < dic[s[i + 1]]:
                ans -= dic[s[i]]
            else:
                ans += dic[s[i]]
        return ans


a = Solution()
print(a.romanToInt("MCMXCIV"))
