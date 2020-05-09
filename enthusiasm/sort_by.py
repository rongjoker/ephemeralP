import sys


def quick_sort(lists, i, j):
    if i >= j:
        return list
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


list1 = [2, 3, 5, 4]
list2 = quick_sort(list1, 0, len(list1) - 1)
for x in list2:
    print(x)


def merge(list_left, list_right):
    """
    入参数组都是有序的，此处将两个有序数组合并成一个大的有序数组
    """
    # 两个数组的起始下标
    l, r = 0, 0

    new_list = []
    while l < len(list_left) and r < len(list_right):
        if list_left[l] <= list_right[r]:
            new_list.append(list_left[l])
            l += 1
        else:
            new_list.append(list_right[r])
            r += 1
    new_list += list_left[l:]
    new_list += list_right[r:]
    return new_list


def merge_sort(mylist):
    """归并排序
    mylist: 待排序数组
    return: 新数组list
    """
    if len(mylist) <= 1:
        return mylist

    mid = len(mylist) // 2
    list_left = merge_sort(mylist[:mid])
    list_right = merge_sort(mylist[mid:])
    return merge(list_left, list_right)


# if __name__ == "__main__":
#     mylist = [12, 33, 199, 0, 54, 33, 11]
#     result = merge_sort(mylist)
#     print(f'归并排序后：{result}')


def fib(num=10):
    a = 1
    b = 1

    count = 0

    while count < num:
        print(a)
        print(b)
        a = a + b
        b = a + b
        count += 1

    print('above is fib')


fib()
