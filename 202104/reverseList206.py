import sys


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head: ListNode) -> ListNode:
    previous = None
    current = head
    while current:
        temp = current.next
        current.next = previous
        previous = current
        current = temp

    return previous


head_x = ListNode(12)
p1 = ListNode(11)
p2 = ListNode(10)
p3 = ListNode(9)
p4 = ListNode(8)
head_x.next = p1
p1.next = p2
p2.next = p3
p3.next = p4
print(reverseList(head_x).val)
