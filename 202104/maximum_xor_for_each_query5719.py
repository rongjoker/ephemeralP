import sys
# 5719. 每个查询的最大异或值 https://leetcode-cn.com/problems/maximum-xor-for-each-query/
from typing import List


class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        lists = []
        product = None
        for n in nums:
            product = product ^ n if product else n
            index = maximumBit - 1
            sumx = 0
            product_ = product
            while index >= 0:
                temp = product_ ^ (1 << index)
                if temp >= product_:
                    sumx = sumx + (1 << index)
                    product_ = temp
                index = index - 1
            lists.append(sumx)
        left = 0
        right = len(lists) - 1
        while left < right:
            temp = lists[left]
            lists[left] = lists[right]
            lists[right] = temp
            left = left + 1
            right = right - 1
        return lists


a = Solution()
print('sumx-reverse', a.getMaximumXor([2, 3, 4, 7], 3))


# 利用自反性和python操作数组的便利性 0 <= nums[i] < 2maximumBit
# 由于 0 <= nums[i] < 2maximumBit 所以每个数字能运算的最大数字就是2maximumBit -1 ，所以就是把数字n ^ (2maximumBit -1 ) 即可
# 也就是n ^ k = (2maximumBit -1 )  => k = n ^ (2maximumBit -1 )
# 异或运算满足交换律和结合律
class Solution2:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        lists = []
        product = 0
        mask = (1 << maximumBit) - 1  # 能左移的最高位置
        for n in nums:
            product ^= n
        while nums:
            lists.append(product ^ mask)
            product ^= nums.pop()
        return lists


a = Solution2()
print('sumx-reverse', a.getMaximumXor([2, 3, 4, 7], 3))
print(1 & 0)
listx = [0 for x in range(0, 10)]
print(listx)
