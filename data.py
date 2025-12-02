# import numpy as np
# import pandas as pd

# x = np.linspace(0, 1, 1000)
# y = []

# for i in x:
#     y.append(np.sin(i))


# df = pd.DataFrame(x, columns=["in"])
# df["out"] = y

# df.to_csv("train.csv", index=False)

def check(n):
    mask = 0
    while(n != 0):
        if (mask & (1 << (n % 10))) != 0:
            return False
        else:
            mask = mask | (1 << (n % 10))
        
        n = n // 10

    return True

n = int(input())

while(True):
    n+=1
    if(check(n)):
        print(n)
        break
