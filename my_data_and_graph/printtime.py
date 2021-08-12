import matplotlib.pyplot as plt

with open("../my_data_and_graph/times.txt", "r") as f:
# with open("../loss.txt", "r") as f:
    content = [eve.strip("\n") for eve in f]
res = []
for eve in content:
    if eve == "----":
        continue

    res.append(float(eve))

plt.figure(1)
plt.plot(content)
# plt.savefig("../my_data_and_graph/graphs/DDQN_CBDDloss-per.5000epi.png")
plt.show()
