import sys
import random

from mediocre import quantity_candidate, quality_candidate

print(' hello world')
a = 1
b = 1

count = 0

while count < 10:
    print(a)
    print(b)
    a = a + b
    b = a + b
    count += 1

print('above is fib')


def provoke(kkk=1000):
    if kkk == 1000:
        return 'is kkk is 1000,it must be default'
    else:
        return 'so it is %d' % kkk


# def provoke(kkk = 1000):
#     print ('is kkk is 1000,it must be default ; so it is %d' %kkk)


print(provoke())
print(provoke(1222))


def i_seek_you(num1, num2=1, num3=2):
    print('num1 is %d' % num1)
    print('num2 is %d' % num2)
    print('num3 is %d' % num3)


i_seek_you(123)
i_seek_you(0, 1)
i_seek_you(2, 3, 4)

print(quantity_candidate.gun_power())
print(quality_candidate.gun_power())
