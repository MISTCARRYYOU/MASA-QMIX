import matplotlib.pyplot as plt

with open("../my_data_and_graph/DDQN_CBDD.loss.txt", "r") as f:
# with open("../loss.txt", "r") as f:
    content = [float(eve.split(",")[0].split("(")[-1]) for eve in f]
plt.figure(1)
plt.plot(content)
plt.savefig("../my_data_and_graph/graphs/DDQN_CBDDloss-per.5000epi.png")
plt.show()
