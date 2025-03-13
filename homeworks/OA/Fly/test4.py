import re

def is_email(email):
    reg = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9_-]@[a-zA-Z0-9_-][a-zA-Z0-9._-]*[a-zA-Z0-9]$"
    return re.match(reg, email) != None
    
print(is_email("abc@d.com")) # True
print(is_email("abc@.com")) # False
print(is_email("abc@com")) # False
print(is_email("abc@.com")) # False
print(is_email(".abc@com.")) # False
print(is_email("abc@qq.com.cn")) # True