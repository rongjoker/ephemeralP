import sys


class MinStack:
    def __init__(self):
        self.lists = []

    def push(self, x: int) -> None:
        if len(self.lists) > 0:
            self.lists.append([x, min(self.lists[len(self.lists) - 1][1], x)])
        else:
            self.lists = [[x, x]]

    def pop(self) -> None:
        del self.lists[len(self.lists) - 1]

    def top(self) -> int:
        return self.lists[len(self.lists) - 1][0]

    def min(self) -> int:
        return self.lists[len(self.lists) - 1][1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
