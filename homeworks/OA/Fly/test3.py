# # 找到一个【没有完全平方数的数列】的第x项

# squares = set([i**2 for i in range(1, 1000)])

# def find_xth_item(x):
#     i = 1
#     while True:
#         if i not in squares:
#             x -= 1
#             if x == 0:
#                 return i
#         i += 1
    

# print(find_xth_item(1)) # 2
# print(find_xth_item(2)) # 3
# print(find_xth_item(2023)) 

class MyClass:
    def __new__(cls, arg):
        obj = super().__new__(cls)
        obj.arg = arg + 1
        return obj
    def __init__(self, arg):
        self.arg += arg  # 进一步初始化，修改 arg

obj = MyClass(2)
print(obj.arg)


