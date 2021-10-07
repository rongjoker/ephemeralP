import sys


def quick_sort(lists, i, j):
    if i >= j:
        return lists
    pivot = lists[i]
    low = i
    high = j
    while i < j:
        while i < j and lists[j] >= pivot:
            j -= 1
        lists[i] = lists[j]
        while i < j and lists[i] <= pivot:
            i += 1
        lists[j] = lists[i]
    lists[j] = pivot
    quick_sort(lists, low, i - 1)
    quick_sort(lists, i + 1, high)
    return lists


# 借助python 的数组特性来实现快排，代码量最小
def quick_sort_simple(arr):
    n = len(arr)
    # 长度小于等于1说明天然有序
    if n <= 1:
        return arr

    # pop出最后一个元素作为标杆
    mark = arr.pop()
    # 用less和greater分别存储比mark小或者大的数
    less, greater = [], []
    for x in arr:
        if x <= mark:
            less.append(x)
        else:
            greater.append(x)
    arr.append(mark)
    return quick_sort_simple(less) + [mark] + quick_sort_simple(greater)


# list1 = [2, 3, 5, 4]
# list2 = quick_sort(list1, 0, len(list1) - 1)
# for x in list2:
#     print(x)

# 该方法为对数据进行一次排序，使数组分成左侧小右侧大的两组数据
# j 为分割点，j的左侧比pivot小，j的右侧比pivot大
# 初始情况下j为0,迭代下标index为0
# 数组开始迭代，遇到大于pivot的，不变化，继续迭代；遇到小于pivot的，将j点的数据和index点的数据交换，同时j向前推进一位（相当于原来的j点安置了小于pivot的数据）
def sort_sorted_array(array):
    j, index, pivot = 0, 0, array[len(array) - 1]
    while index < len(array) - 1:
        temp = array[index]
        if temp < pivot:
            array[index] = array[j]
            array[j] = temp
            j += 1
        index += 1
    array[index] = array[j]
    array[j] = pivot
    print(f'j:{j}')
    return array


# 快排的本质是在sort_sorted_array方法的基础上继续迭代，sort_sorted_array会将数组以中轴分为左小右大，那么以中轴分割2个数组，
# 再分别sort_sorted_array，递归到不可再产生中轴为止
def mediocre_quick_sort(array, start, end):
    print(f'start:{start};end:{end}')
    if start >= end:
        return array

    j, index, pivot = 0, 0, array[end]
    while index < end:
        temp = array[index]
        if temp < pivot:
            array[index] = array[j]
            array[j] = temp
            j += 1
        index += 1
    array[index] = array[j]
    array[j] = pivot
    print(f'j:{j};start:{start};end:{end}')

    mediocre_quick_sort(array, start, j-1)
    mediocre_quick_sort(array, j+1, end)
    return array


a = [77, 1, 3, 44, 23, 2, 88, 7, 9, 13, 4, 6]
# a = [1, 3, 2, 4]
print(mediocre_quick_sort(a, 0, len(a) - 1))

# print(sort_sorted_array([1, 2, 3, 4]))
# print(sort_sorted_array([1, 2, 3, 4]))
# print(sort_sorted_array([1, 3, 2, 4]))
# print(sort_sorted_array([1, 3, 2, 4]))
# print(sort_sorted_array([1, 3, 2, 4]))


# print(quick_sort_simple([1, 3, 2, 4]))
