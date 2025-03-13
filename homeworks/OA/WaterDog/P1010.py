import math
n = int(input())

def fuck(x):
    if x == 0:
        return "0"
    power = math.floor(math.log2(x))
    remain = x - 2 ** power
    result = "2"
    if power != 1:
        result += "(" + fuck(power) + ")"
    if remain > 0:
        result += "+"
        result += fuck(remain)
    return result

print(fuck(n))