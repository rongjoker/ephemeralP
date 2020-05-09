
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def reverse_list(head: ListNode):
    pre = None
    cur = head
    # 遍历链表，while循环里面的内容其实可以写成一行
    # 这里只做演示，就不搞那么骚气的写法了
    while cur:
        # 记录当前节点的下一个节点
        tmp = cur.next
        # 然后将当前节点指向pre
        cur.next = pre
        # pre和cur节点都前进一位
        pre = cur
        cur = tmp
    return pre








