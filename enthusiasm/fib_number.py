

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