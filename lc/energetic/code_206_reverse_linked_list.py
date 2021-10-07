class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def reveal(node: ListNode):
    while node:
        print(node.val)
        node = node.next


# 迭代
def reverse(first: ListNode):
    current = first
    pre = None
    while current:
        temp = current.next
        current.next = pre
        pre = current
        current = temp
    return pre


# 递归
def reverse2(first: ListNode):
    current = first
    pre = None
    while current:
        temp = current.next
        current.next = pre
        pre = current
        current = temp
    return pre


head_x = ListNode(12)
p1 = ListNode(11)
p2 = ListNode(10)
p3 = ListNode(9)
p4 = ListNode(8)
head_x.next = p1
p1.next = p2
p2.next = p3
p3.next = p4

reveal(head_x)
reveal(reverse(head_x))
