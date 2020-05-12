import sys


# 默认循环方法，超出时间限制
def twoSum(nums, target: int):
    length = len(nums)
    a = 0
    while a < length:
        b = a + 1
        while b < length:
            if nums[a] + nums[b] == target:
                return [a, b]
            b += 1
        a += 1


# 用dictionary处理
def twoSum2(nums, target: int):
    dicts = {}
    i = 0
    for num in nums:
        dicts[num] = i
        i += 1
    length = len(nums)
    a = 0
    while a < length - 1:
        if (target - nums[a]) in dicts and a != dicts[target - nums[a]]:
            return [a, dicts[target - nums[a]]]
        a += 1


# 用dictionary处理，优化性能，排除掉本身,边判断边插入dictionary
def twoSum3(nums, target: int):
    dicts = {}
    length = len(nums)
    a = 0
    while a < length:
        if (target - nums[a]) in dicts:
            return [a, dicts[target - nums[a]]]
        dicts[nums[a]] = a
        a += 1


nums_pre = [3, 2, 4]
print(nums_pre)
print(twoSum3(nums_pre, 6))
