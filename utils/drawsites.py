import matplotlib.pyplot as plt

from utils.site import Sites

sites = Sites()

pos = sites.sites_position
x = [eve[0] for eve in pos]
y = [eve[1] for eve in pos]


def pltcircle(x_c, y_c, r, color):
    circle = plt.Circle((x_c, y_c), r, color=color, fill=False)
    plt.gcf().gca().add_artist(circle)


plt.figure(figsize=(20, 8))
plt.xlim([0,50])
plt.ylim([0, 30])
plt.scatter(x, y, label="sites")
for i in range(len(x)):
    plt.text(x[i], y[i], str(i))

# 圈出保障资源覆盖范围
pltcircle(1, 1, 1, "r")

plt.legend()
plt.show()
