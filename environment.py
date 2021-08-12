"""
这个文件主要是表示航母甲板上的站位、保障资源等，以及其奖励函数等
"""
from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes
from utils import util
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
# 整个环境类
class ScheduleEnv(gym.Env):
    environment_name = "Boat Schedule"

    def __init__(self):
        # 类变量的声明
        self.sites = []  # 战位 index索引
        self.jobs = []  # 保障资源 index索引
        self.task = []  # 任务 index索引
        self.planes_obj = Planes()  # 创建总飞机类
        self.planes = []  # 飞机 index索引
        self.state = [[]]  # 状态，环境返回的状态是全局状态
        self.done = False
        self.state_left_time = []  # 战位状态剩余时间
        self.episode_time_slice = []  # 每个step消耗时间组成的episode的时间列表
        self.plane_speed = 0  # 运行速度
        self.initialize()  # 初始化参数
        # 参与dqn决策的plane不需要等待动作，一定会选择一个合适的动作
        # 0-17 代表下一步前往的战位， 18代表由于资源冲突需要等待，19代表处于正忙（加工）动作，20代表已经完成了动作，19、20均不参与训练
        self.action_space = spaces.Discrete(len(self.sites)+3)  # 此时已经初始化完成了，多一维表示什么也不做
        self.id = "Boat Schedule"
        # 下面两个参数还不知道什么意思
        self.reward_threshold = -1000
        self.trials = 50  # 这个就类似于steps

        self.job_record_for_gant = []  # 用于存储调度中间过程四元组

        self.sites_state_global = None  # this para is utilized to indicate the current idle sites and their processing jobs

        # 一个全局状态，一个观测
        self.state4marl = None  # 维护全局state的变量
        self.obs4marl = None

    def initialize(self):
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes()
        # 0-17 共18个战位，每个战位有自己的保障资源、位置，用id索引
        self.sites = sites_obj.sites_object_list
        # 0-8 共9个保障资源，有自己的时间
        self.jobs = jobs_obj.jobs_object_list
        # 任务，里面是任务的序列
        self.task = task_obj.simple_task_object
        # 飞机 0-7
        self.planes = self.planes_obj.planes_object_list
        # 状态 目前18个战位的状态，包括是否被占用了，状态是agent和环境共同的状态，而不是某一个的状态，包括战位状态、飞机状态、战位资源状态
        # 0号位置：9 为空闲状态 0-7为目前战位包含的飞机编号，目前默认一个战位最多只能有一个飞机，目前只选择战位，还没有进行资源的选择过程
        # 飞机到达战位之后，默认选择当前待完成任务的下一个任务
        # 1号位置：
        self.state = [[9, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]] for i in range(len(self.sites))]

        self.sites_state_global = [-1 for i in range(len(self.sites))] # -1代表没有被安排保障任务

        self.job_record_for_gant = []  # 用于存储调度中间过程四元组

        # 是否所有飞机都完成了保障作业
        self.done = False
        # 状态的占用时间倒计时，为0时代表这个战位可以被安排下一个job了
        self.state_left_time = np.array([0 for i in range(len(self.sites))])
        self.episode_time_slice = []
        self.plane_speed = self.planes_obj.plane_speed  # 运行速度
        # print("the environment is initialized now !!")
        self.obs4marl = [[] for i in range(len(self.planes))]
        self.current_finishing_jobs = 0
        self.step_count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.initialize()  # 初始化参数
        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        self.jobs[self.planes[i].left_job[0].index_id].time_span,
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0
                       else [
                9,
                0,
                len(self.planes[i].left_job)
            ] for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        # print(info)
        # print(len(state))
        return state  # 151

    # 利用info中的信息整合成为state的信息
    # 以战位为单位进行整合，用掩码的方式选择飞机的信息
    def conduct_state(self, info):
        res = []
        temp = []
        # a1 = 0
        # a2 = 0
        # a3 = 0
        # 开始增加全部站位是否空闲的状态
        for eve in info["sites"]:
            # res.append(eve[1])
            temp.append(eve[1])
            # a1 += 1
        # 开始增加全部站位的当前资源情况
        for eve in info["sites"]:

            res += eve[2]
        #     # a2 += len(eve[2])
        # 开始增加全部飞机的状态信息
        for i, eve in enumerate(info["planes"]):
            res += [eve[2]]
            temp_obs = []
            for l, eve_1 in enumerate(temp):
                if self.planes[i].left_job == []:
                    temp_obs.append(0)
                else:
                    if eve_1 == 9 and self.planes[i].left_job[0].index_id in self.sites[l].resource_ids_list:

                        temp_obs.append(util.count_path_on_road(self.planes[i].position, self.sites[l].absolute_position, self.plane_speed)/40)  # 代表处于空闲状态,下一步飞机可以去
                    else:
                        temp_obs.append(0)  # 否则处于加工状态，下一步飞机不能去
            self.obs4marl[i] = temp_obs + [eve[0], eve[2], eve[1]]

            # self.obs4marl[i] = temp + eve  # 具体每个飞机的观测
            # a3 += 1

        # 更新每个飞机的当前是否处于加工状态
        current_working_plane_ids = []
        for eve in self.state:
            if eve[0] != 9:
                current_working_plane_ids.append(eve[0])
        if current_working_plane_ids == []:
            for k in range(len(self.obs4marl)):
                self.obs4marl[k].append(0)
        else:
            for k in range(len(self.obs4marl)):
                if k in current_working_plane_ids:
                    # 正忙状态下的观测也进行处理为除了最后三位都是0
                    self.obs4marl[k] = [0 if kk < len(self.obs4marl[k])-2 else self.obs4marl[k][kk] for kk in range(len(self.obs4marl[k]))]
                    self.obs4marl[k].append(1)
                else:
                    self.obs4marl[k].append(0)
        obslen = len(self.obs4marl[0])
        zero_obs = [0 for i in range(obslen)]

        # 将加工完成的agent的观测设置为0
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:  # 代表第i个飞机目前已经加工完成了
                self.obs4marl[i] = zero_obs

        # res.append(0)  # 最后一位为飞机id的标识符号，需要在训练中被替换
        # print(len(res), a1, a2,a3)
        self.state4marl = np.array(res)
        return np.array(res)

    # 去掉动作中选择重复的状态，这个在网络部分是软限制
    # 但是18是可以重复的
    def check_inflict_action(self, action):
        res = []

        for eve in action:
            if eve == len(self.sites) or eve == 19 or eve == 20:
                res.append(eve)
            else:
                if eve not in res:
                    res.append(eve)
                else:
                    raise Exception("sloppy error in actions", action)
                    # assert False
        return res

    # 进行动作的替换
    def action_replace(self, action):
        res = []
        real_conflict_num = 0
        for eve in action:
            if eve == 20 or eve == 19:
                res.append(18)
            else:
                if eve == 18:
                    real_conflict_num += 1

                res.append(eve)
        return res, real_conflict_num

    # action是 [-1, 0, 1, 2, 3, ...] 表示每架飞机选择的战位-1表示不选择继续处于空闲状态，动作的选择都是合法的，但是动作不知道这个战位有没有需要的资源
    # 在所有action里第一个完成之后就发生一次step，时间这里如何安排？--
    def step(self, action):
        self.step_count += 1
        # print("开始交互了")
        action, real_conflict_num = self.action_replace(action) # 将action中的20换成18
        # if real_conflict_num != 0:
        #     print(real_conflict_num)
        count_break_rules = 0
        # print(action)
        assert len(action) == len(self.planes)
        rewards = [0 for eve in action]
        max_time_on_roads = [0 for eve in action]
        count_for_reward = 0
        action = self.check_inflict_action(action)
        time_span_increase = np.array([0 for eve in self.sites])  # 用于接下来存储这个step下由于action带来的未来各个位置的时间增加
        for i, site_id in enumerate(action):  # i代表了飞机的id
            if site_id == len(self.sites):  # 代表这个飞机不安排保障任务
                pass
            else:  # 安排保障任务
                if self.planes[i].left_job[0].index_id in self.sites[site_id].resource_ids_list:  # 如果选择的战位有需要的保障资源
                    # time_on_road为0代表其留在了原地加工
                    time_on_road = util.count_path_on_road(self.planes[i].position,
                                                           self.sites[site_id].absolute_position.tolist(), self.plane_speed)
                    # if time_on_road == 0:
                    #     print(action, i, self.planes[i].position, self.sites[site_id].absolute_position.tolist())
                    # 默认执行下一个待执行的job
                    # self.planes[i]
                    # self.sites[site_id]
                    # print(self.planes[i].left_job)
                    # 已经完成任务的飞机仍然在被调度动作！！
                    # 执行动作
                    if type(site_id) == int:
                        self.save_env_info(
                            (sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id, i))
                    else:
                        self.save_env_info((sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id.item(), i))
                    temp_time = self.planes[i].execute_task(self.planes[i].left_job[0], self.sites[site_id])

                    time_span_increase[site_id] = temp_time + time_on_road
                    # self.state[site_id][0] = i  # 表示这个站位已经被占据了
                    count_for_reward += 1
                    # 为构造每个飞机的reward存储maxtime，方便归一化
                    max_time_on_roads[i] = time_on_road

                    # 更新sites_state_global
                    # self.sites_state_global[site_id] = self.planes[i].left_job[0].index_id
                    # print(self.sites_state_global)

                else:
                    raise Exception("不合理的动作没有mask", self.sites_state_global, i, site_id,action,self.planes[i].left_job[0].index_id,
                                    self.sites[site_id].resource_ids_list, self.state)

        real_did = 0
        for eve in action:
            if eve < 18:
                real_did += 1

        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # 代表这个飞机不安排保障任务
                # 这里有问题，这是在鼓励模型学习尽可能给飞机安排调度任务
                # 增加对无效动作的惩罚：这两个其实是规则，这两个环境都没法做，因此返回惩罚
                #     1、选择了重复动作（已经在状态战位被占据的情况下人选择这个战位）
                #     2、选择的战位没有下一步所包含的资源-->这个过程的状态表示其实比较隐晦
                rewards[i] = - 30  # 只传入因为资源冲突而等待的地方
            else:  # 安排保障任务
                if rewards[i] == 0:
                    # rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)-real_conflict_num
                    rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)
                    # rewards[i] = - 0.5*len(self.planes[i].left_job)
                    # if rewards[i] == -1:
                    #     with open("templook.txt", "a") as f:
                    #         print(max_time_on_roads[i], max(max_time_on_roads), file=f)
                else:
                    pass

        # 开始更新当前的状态剩余时间
        self.state_left_time = self.state_left_time + time_span_increase
        # 找到更新完后非0的最小时间，因为有的战位处于空闲状态

        # print(self.state_left_time, )

        min_time = util.min_but_zero(self.state_left_time)
        # print("haoshi：", min_time)
        self.episode_time_slice.append(min_time)  # 这个step消耗的时间
        self.state_left_time = util.advance_by_min_time(min_time, self.state_left_time)  # step推进

        # 更新状态,主要是检查哪些状态用完了
        # state transition 2
        for i, eve_time in enumerate(self.state_left_time):
            if eve_time == 0:
                self.sites_state_global[i] = -1  # 更新做完的sites
                self.sites_obj.update_site_resources(self.sites_state_global)
                self.state[i][0] = 9
                self.state[i][1] = [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]
            else:
                assert self.state[i][0] != 9  # 不为9的一定被占据了

        # 判断当前episode是否完成了
        is_all_done = [-1 for eve in self.planes]
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                is_all_done[i] = 0
        # self.current_finishing_jobs = sum(is_all_done) + len(is_all_done) - self.current_finishing_jobs
        if sum(is_all_done) == 0:
            self.done = True
        else:
            self.done = False

        # 更新reward 注意reward应该是负数,但是他这个marl环境中的reward是正的
        # 按照飞机为单位更新reward
        left_jobs, all_jobs = self.planes_obj.count_jobs()
        if self.done:
            reward = 6000 / (sum(self.episode_time_slice) + max(self.state_left_time))
            # print(11, reward)

        else:
            reward = real_did - self.step_count/60 - real_conflict_num*2
            # print(22, reward)
        # reward = -real_conflict_num

        # print("总剩余工作数：", left_jobs)
        # reward = - min_time/(count_for_reward + 2)  # 单位时间内完成的工作越多越好
        # reward = -1

        # reward = -1  # 正在进行的任务数/当前阶段时间

        # 构建info，形成非结构化的数据供agent进一步抽象，不直接使用self.planes和self.sites是为了方便数据整理
        # info:  # 最里层都是三维的列表
        # {
        #     "战位"：[[[位置], 空闲与飞机编号数字, [保障资源列表]], [], [], [], ...]
        #     "飞机"：[[下一个要完成的任务，完成任务数，剩余工作数], [], [], ...]
        # }
        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0
                       else [
                        9,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                    ]for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        # print("min_time:", min_time)
        # print("left_time:", self.state_left_time)
        return reward, self.done, {"time": sum(self.episode_time_slice)+max(self.state_left_time),
                                          "left": self.state_left_time,
                                          "original_state": self.state,
                                          "planes_obj": self.planes,
                                          "rewards": rewards,
                                          "count_break_rules": count_break_rules,
                                          "sites_state_global": self.sites_state_global,
                                   "episodes_situation": self.job_record_for_gant
                                   }

    # 根据飞机的id返回该飞机agent可行的动作集合
    # 注意需要保证从0开始调的
    # 注意返回值有两种情况，分别是字符串和正常情况
    # 还需要判断这个飞机现在是不是处于空闲状态
    # return -> 0: 飞机完成了调度任务 1: 飞机处于非空闲状态 list: 正常可调运状态
    def get_avail_agent_actions(self, agent_id):
        # 检查飞机是否处于正忙状态
        for eve in self.state:
            if agent_id == eve[0]:  # 代表此飞机还在处于加工状态
                # return [0 for i in range(18)] + [1]  # 1
                return [0 for i in range(18)] + [0, 1, 0]
        # 如果飞机准备进行下一步操作则执行下部分程序
        res = [0 for eve in self.sites_state_global]
        for i, eve in enumerate(self.sites_state_global):
            if eve == -1:
                if len(self.planes[agent_id].left_job) != 0:
                    # 判断该飞机下一个要完成的任务是否被包含在了资源列表中
                    if self.planes[agent_id].left_job[0].index_id in self.sites[i].resource_ids_list:
                        res[i] = 1
                else:  # 证明此时的这个飞机已经完成了所有的调度计划
                    # return [0 for i in range(18)] + [1]  # 0
                    return [0 for i in range(18)] + [0, 0, 1]
        return res + [1, 0, 0]

    # 每次一个plane选择完动作后都调用以下这个函数用于更新环境状态
    # state transition 1
    def has_chosen_action(self, action_id, agent_id):
        assert self.planes[agent_id].left_job != []
        # print(action_id, agent_id)
        self.sites_state_global[action_id] = self.planes[agent_id].left_job[0].index_id  # 更新战位状态信息
        # 更新总资源列表的状态-这块不能马虎，注意这里是在选择合理的动作而不是已经做了动作
        self.sites_obj.update_site_resources(self.sites_state_global)
        self.state[action_id][0] = agent_id  # 表示这个站位已经被占据了,更新状态
        self.state[action_id][1] = [1 if j in self.sites[action_id].resource_ids_list else 0 for j in range(9)]  # 更新资源抢占状态

    # 每次调用这个函数就存储环境的信息
    # （开始时间、 持续时间（task名称）、 在哪个资源加工的、 加工的飞机名称）
    def save_env_info(self, job_transition):
        self.job_record_for_gant.append(job_transition)

    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    # 返回的是一个列表存储所有的agent观测
    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        agents_obs = [self.get_obs_agent(i) for i in range(len(self.planes))]
        return agents_obs

    # 返回的是每一个agent的观测
    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    # 获得环境的相关信息
    def get_env_info(self):
        return {
            "n_actions": len(self.sites) + 3,  # 还是得把空闲动作加上去
            "n_agents": len(self.planes),
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]),
            "episode_limit": 80  # 注意，如果在80的长度内无法完成调度工作的话程序会报错，但是设置太大后面全是paddings
        }
