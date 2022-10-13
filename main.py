
import numpy as np
import pickle
from environment import ScheduleEnv
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from utils.PDRs.shortestDistence import SDrules

np.random.seed(2)

# game_scores, rolling_scores的区别，一个是整个episode的reward
# ，一个是一个trial的rewad，两个指标都不参与训练，trial是人为指定的一个固定长度的n个step的过程


# 强化学习决策函数，带入来自DRL的强化学习agent
def marl_agent_wrapper():

    with open("./my_data_and_graph/historydata/accumulated_rewards.txt", "w") as f:
        print("----", file=f)
    with open("my_data_and_graph/times.txt", "w") as f:
        print("----", file=f)

    with open("./my_data_and_graph/historydata/havealook.txt", "w") as f:
        pass
    with open("./my_data_and_graph/historydata/loss.txt", "w") as f:
        pass
    with open("./my_data_and_graph/historydata/scheduleresults.txt", "w") as f:
        pass

    # import datetime, os, time
    # from shutil import copyfile
    # if os.path.exists("my_data_and_graph/marl.time_reward.txt"):
    #     tar = "my_data_and_graph/marlhisrtorydata/" + str(datetime.date.today()) + "-" + str(time.time()).split(".")[
    #         0] + "marl.time_reward.txt"
    #     with open(tar, "w") as f:
    #         pass
    #     copyfile("my_data_and_graph/marl.time_reward.txt", tar)
    #
    # with open("my_data_and_graph/marl.time_reward.txt", "w") as f:
    #     pass
    # for i in range(8):  # 因为一共8种marl算法
    args = get_common_args()

    if args.alg.find('coma') > -1:  # 判断模型的参数
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # 加载调度环境
    env = ScheduleEnv()

    env.reset()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    print("是否加载模型（测试必须）：", args.load_model, "是否打印中间变量：", args.havelook, "是否训练：",args.learn)
    runner = Runner(env, args)

    if args.learn:
        runner.run(0)  # 原来跑多种算法，run传入的是算法的id
    else:
        _, reward = runner.evaluate()
        print('The ave_reward of {} is  {}'.format(args.alg, reward))


# 随机决策函数，用于测试环境
def random_agent_wrapper():

    episodes = 50

    env = ScheduleEnv()
    temp_save = [0]
    EATs = []
    schedule_processes = []
    for episode in range(episodes):

        s = env.reset()
        is_terminal = False
        while not is_terminal:
            # print(1)
            actions = []
            # 每次只调运非空闲的agent
            temp_not_idle_agents = []
            for m in range(len(env.sites)):
                if s[m] != 9:
                    temp_not_idle_agents.append(s[m])

            for i in range(len(env.planes)):
                if i in temp_not_idle_agents:  # 证明i正忙着
                    actions.append(18)
                else:
                    # print(i)
                    avail_actions = env.get_avail_agent_actions(i)
                    tem_choose = []
                    if type(avail_actions) != str:
                        for k, eve in enumerate(avail_actions):
                            if eve == 1:
                                tem_choose.append(k)
                        if tem_choose == []:
                            action = 18
                        else:
                            action = np.random.choice(tem_choose, 1, False)[0]
                            env.has_chosen_action(action, i)
                        actions.append(action)
                    else:
                        actions.append(18)
            s, r, is_terminal, dict = env.step(actions)
            # print(actions)
            # print(s[:18])
        EATs.append(dict["time"])
        schedule_processes.append(env.job_record_for_gant)
        print(env.job_record_for_gant)
        print(dict["time"], "-----------------------------------")
    print(sum(EATs)/len(EATs))
    # 存储中间结果
    with open("./my_data_and_graph/pickles/process.pk", "wb") as f:
        pickle.dump(schedule_processes, f)


def SDrules_agent_wrapper():
    EPISODES = 50

    sd_rules = SDrules()
    env = ScheduleEnv()
    sites_locations = env.sites_obj.sites_position

    actions = []
    for episode in range(EPISODES):
        done = False
        env.reset()
        while not done:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(8)
            # agents_id_sequence = sd_rules.MLF_generate_agents_sequence(env.planes)
            # agents_id_sequence = sd_rules.LLF_generate_agents_sequence(env.planes)


            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)
                current_plane_location = env.planes[agent_id].position
                action = sd_rules.choose_action(agent_id, avail_actions, current_plane_location, sites_locations)
                actions.append(action)
                if action < 18:
                    env.has_chosen_action(action, agent_id)
            # print(actions)
            # 按照action的顺序进行重新整理，因为环境需要按照顺序接受actions
            reorder_actions = [-1 for i in range(8)]
            for i in range(8):
                reorder_actions[agents_id_sequence[i]] = actions[i]
            _, done, info = env.step(reorder_actions)
        print(info["time"])
    print(info['episodes_situation'])


if __name__ == "__main__":
    marl_agent_wrapper()
