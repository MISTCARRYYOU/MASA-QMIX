import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="darkgrid")
sns.set()
# 引入数据


def get_res(path):
    with open(path, "r") as file:
        content = [eve.strip("\n") for eve in file]
    res = []
    temp = []
    for eve in content:
        if eve.find("----") != -1:
            res.append(temp)
            temp = []
        else:
            temp.append(float(eve))
    if [] in res:
        res.remove([])
    return res


c1 = get_res("accumulated_rewards.txt")
# c1 = get_res("accumulated_rewards.txt")

# c2 = get_res("times1.txt")
# c3 = get_res("times2.txt")
c = c1
print(c)
x_exp = list(range(len(c1[0])))

x = []
y = []
for eve in c:
    x += x_exp
    y += eve

# RL图
sns.lineplot(x=np.array(x), y=np.array(y))
plt.show()
