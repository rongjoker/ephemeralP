import sys


def merge(list_left, list_right):
    """
    入参数组都是有序的，此处将两个有序数组合并成一个大的有序数组
    l命名容易和1混淆，故不建议使用
    """
    # 两个数组的起始下标
    left0, right0 = 0, 0

    new_list = []
    while left0 < len(list_left) and right0 < len(list_right):
        if list_left[left0] <= list_right[right0]:
            new_list.append(list_left[left0])
            left0 += 1
        else:
            new_list.append(list_right[right0])
            right0 += 1
    new_list += list_left[left0:]
    new_list += list_right[right0:]
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


# 排列2个有序数组
def sort_sorted_array(a, b):
    c = []
    i, j = 0, 0
    while i < len(a) or j < len(b):
        if i == len(a):
            c.append(b[j])
            j += 1
            continue
        elif j == len(b):
            c.append(a[i])
            i += 1
            continue
        elif a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    return c


# a = [1, 4, 6, 7, 9]
# b = [2, 4, 5]


# print(sort_sorted_array(a, b))


""" 
归并排序的逻辑，可以分为2个过程
第一个过程，是将2个有序数组进行合并，给2个数组分为添加游标，逐个比较大小，将小的那条数据加入到新的数组并将那条数据所在的原始数组游标加1，最终迭代完成2个数组
第二个过程，是为第一个过程制造有序数组，当一个数组只有一条数据的时候，它就是有序的，所以将数组不断中分到最小单位为1，即产生有序数组
在实际操作中，通过递归将1和2两个过程依次不停地执行，一直到递归结束
递归基为数组大小为1，即结束切割
"""


def mediocre_merge_sort(x):
    if len(x) <= 1:
        return x
    n = len(x) // 2
    print(f'x:{x};n:{n}')
    a, b = mediocre_merge_sort(x[: n]), mediocre_merge_sort(x[n:])
    print(f'切分到左右：进行排序 x:{x};a:{a};b:{b}')
    len_a, len_b = len(a), len(b)
    c = []
    i, j = 0, 0
    while i < len_a or j < len_b:
        if i == len_a:
            c.append(b[j])
            j += 1
            continue
        elif j == len_b:
            c.append(a[i])
            i += 1
            continue
        elif a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    return c
    # print(x[2:4]) 左闭右开


def spit():
    m = [1, 2]
    n = len(m) // 2
    xxx = m[: n]
    print(len(xxx))


k = [1, 12, 31, 4, 6, 3, 7, 9, 11, 3, 9]
print(mediocre_merge_sort(k))

# spit()
