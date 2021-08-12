"""
优先选择离自己最近并且有资源的进行加工
都是优先选择距离最近的，但是选择的顺序不一样，有的是fifo（for循环），有的是按照剩余任务最多进行选择，等等。。。
"""
from utils.util import count_path_on_road
import numpy as np

class SDrules:
    def __init__(self):
        pass

    # avail: [1,2,...], current: (1,2), sites: [(1,2), (2,3),...]
    # 这个函数直接按照可行动作中距离最短的进行选择动作
    def choose_action(self, agent_id, avail_actions, current_plane_location, sites_loactions_fixed):
        avail_ids = []
        for i, eve in enumerate(avail_actions[:-3]):
            if eve == 1:
                avail_ids.append(i)
        if avail_ids == []:
            for i, eve in enumerate(avail_actions[-3:]):
                if eve == 1:
                    return i + 18
            raise Exception("available actions error!", avail_actions)
        # 否则avail_ids不为空
        distances = []
        for eve_site_id in avail_ids:
            distances.append(count_path_on_road(current_plane_location, sites_loactions_fixed[eve_site_id], 20))
        distances_array = np.array(distances)
        arg_min = np.argsort(distances_array)[0]
        return avail_ids[arg_min]  # 返回其最短路径的规则

    # 就是正常顺序
    def FIFO_generate_agents_sequence(self, num):
        return [i for i in range(num)]

    # 按照谁剩的多谁优先的顺序
    def MLF_generate_agents_sequence(self, planes_objs_list):
        left_jobs = []
        for eve_plane in planes_objs_list:
            left_jobs.append(len(eve_plane.left_job))
        left_jobs_array = np.array(left_jobs)
        arg_sort = np.argsort(-left_jobs_array)
        return list(arg_sort)

    # 按照谁剩的少谁优先的顺序
    def LLF_generate_agents_sequence(self, planes_objs_list):
        left_jobs = []
        for eve_plane in planes_objs_list:
            left_jobs.append(len(eve_plane.left_job))
        left_jobs_array = np.array(left_jobs)
        arg_sort = np.argsort(left_jobs_array)
        return list(arg_sort)
