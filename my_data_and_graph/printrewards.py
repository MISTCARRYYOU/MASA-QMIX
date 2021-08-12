import matplotlib.pyplot as plt

with open("../my_data_and_graph/DDQN_CBDD.score.txt", "r") as f:
# with open("../loss.txt", "r") as f:
    content = [round(float(eve), 2) for eve in f]
plt.figure(1)
plt.plot(content)
plt.savefig("../my_data_and_graph/graphs/DDQN_CBDD-per.5000epi-rewards.png")
plt.show()
