# Question 1

class MyClass:
    class_variable = 0
    def __init__(self, instance_variable):
        self.class_variable = instance_variable
    
    @classmethod
    def class_method(cls):
        cls.class_variable += 1
    
    @staticmethod
    def static_method():
        return MyClass.class_variable
    
obj1 = MyClass(1)
obj2 = MyClass(2)
MyClass.class_method()
print(obj2.static_method(), obj1.class_variable, obj2.class_variable)

# Answer: 1 1 2
# Explanation: class_variable is a class variable, so it is shared by all instances of the class.
# obj1.class_variable is an instance variable, so it is unique to obj1.
# obj2.class_variable is an instance variable, so it is unique to obj2.


# Question 3

class Animal:
    def __init__(self, color):
        self.__color = color
    
    @property
    def color(self):
        return self.__color
    
    # Fill in the blank
    def color(self, color):
        self.__color = color

# Answer: @color.setter
# Explanation: The color method is a getter method. To create a setter method, we need to use the @property decorator followed by the setter method name.
        
        
# Question 4

def sort_string(s):
    words = s.split()
    sorted_words = sorted(words, key=lambda x: (len(x), x), reverse=True)
    return ' '.join(sorted_words)

print(sort_string("This is a test string"))

# Answer: string test This is a
# Explanation: The input string is split into words, sorted by length and then lexicographically, and then joined back together.
# Length: "This" (4), "is" (2), "a" (1), "test" (4), "string" (6) -> "string", "This", "test", "is", "a"
# Lexicographical order: "This" < "test" because "T" < "t"
# Final output: "string test This is a"