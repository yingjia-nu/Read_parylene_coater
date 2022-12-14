import random
import string


lower = random.choices(string.ascii_letters + string.digits, k=12)
print(lower)