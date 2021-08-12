"""
这里算是agent与environment的耦合文件
"""
from environment import ScheduleEnv
import agent


def train():
    episode_len = 500
    episodes = 1
    env = ScheduleEnv()
    for i in range(episodes):
        state, info = env.reset()

        for step in range(episode_len):
            print("------------------step {}------------begin".format(step))
            action = agent.random_decision_agent(state, info)
            state, reward, done, info = env.step(action)
            print("action:", action)
            print("reward, state, done:", reward, state, done)
            print("------------------step {}------------over".format(step))
            if done is True:
                print("总时间：", sum(env.episode_time_slice) + max(env.state_left_time))  # 应该是总时间+最大的剩余时间
                break


if __name__ == "__main__":
    train()

