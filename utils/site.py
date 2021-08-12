"""
战位的相关信息，被环境调用
A-R 20个战位
"""
import numpy as np
from utils.job import Jobs
import copy


# 所有战位的类
class Sites:

    def __init__(self):

        # 所有战位对象
        # id: 0 - 17
        self.sites_object_list = []
        sites_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", 'O', 'P', 'Q', 'R']
        sites_positions = [
            [40, 13.5], [38, 14.5], [36, 15], [34, 15.8], [32, 16.4], [30, 17.1],
            [6, 16.2], [4.2, 14], [3.6, 11.5], [3.1, 9.3], [7, 8.4], [11, 7.6],
            [15, 6.6], [19, 5.4], [28.7, 17.9], [27.7, 19.2], [26.7, 20.6], [24.7, 20.8]
        ]
        self.sites_position = sites_positions
        # 每个战位拥有的保障资源
        # A-F
        # G-N
        # O-R
        sites_resources_range = [
            "all", "all", "all", "all", "all", "all",
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6],
            [0, 1, 7, 2, 8], [3, 4, 5, 6], [0, 1, 7, 2, 8], [3, 4, 5, 6]
        ]
        for i in range(len(sites_codes)):
            if sites_resources_range[i] == "all":
                temp_object = Site(i, sites_positions[i], list(range(0, 9, 1)))
            else:
                temp_object = Site(i, sites_positions[i], sites_resources_range[i])
            self.sites_object_list.append(temp_object)

        # 增加资源抢占的约束
        # 0-3号战位一对一保障服务，
        # 4和5共用一套，
        # 6-7-8-9,10-11-12-13,14-15-16-17共用一套，
        # 共9*4+9*1+9*3=72个保障点
        self.restrict_dict = {
            0: {},
            1: {},

            2: {},
            3: {},

            4: {5: [0,1,2,3,4,5,6,7,8]},
            5: {4: [0,1,2,3,4,5,6,7,8]},

            6: {8: [0, 1, 7, 2, 8]},
            7: {9: [3, 4, 5, 6]},
            8: {6: [0, 1, 7, 2, 8]},
            9: {7: [3, 4, 5, 6]},

            10: {12: [0, 1, 7, 2, 8]},
            11: {13: [3, 4, 5, 6]},
            12: {10: [0, 1, 7, 2, 8]},
            13: {11: [3, 4, 5, 6]},

            14: {16: [0, 1, 7, 2, 8]},
            15: {17: [3, 4, 5, 6]},
            16: {14: [0, 1, 7, 2, 8]},
            17: {15: [3, 4, 5, 6]}
        }

        # self.restrict_dict = {
        #     0: {},
        #     1: {},
        #     2: {},
        #     3: {},
        #
        #     4: {},
        #     5: {},
        #
        #     6: {},
        #     7: {},
        #     8: {},
        #     9: {},
        #
        #     10: {},
        #     11: {},
        #     12: {},
        #     13: {},
        #
        #     14: {},
        #     15: {},
        #     16: {},
        #     17: {}
        # }



    # 每次查找战位resources之前都更新一下战位的即时资源，根据资源约束关系，至于资源抢占是如何分配的，那就是先到先得
    # 输入:
    # temp_sites_state_global: [-1,-1,-1,-1,...,0,8,2,3] 是资源id
    def update_site_resources(self, temp_sites_state_global):
        # 按照当前飞机占用的战位进行更新
        for i, eve in enumerate(temp_sites_state_global):
            if eve == -1:
                pass
            else:
                conflict_sites = list(self.restrict_dict[i].keys())
                if len(conflict_sites) == []:
                    continue
                for each_con_site in conflict_sites:
                    assert eve in self.restrict_dict[i][each_con_site]
                    temp = copy.deepcopy(self.restrict_dict[i][each_con_site])
                    temp.remove(eve)
                    assert len(temp) > 2  # 每次只会有一个被移除
                    self.sites_object_list[each_con_site].update_resorces(temp)


# 每一个战位的类
class Site:
    def __init__(self, site_id, relative_position, resource_ids_list):
        self.site_id = site_id
        self.absolute_position = np.array([10, 10]) + np.array([20*relative_position[0], 20*relative_position[1]])
        self.resource_jobs = Jobs()
        self.resource_jobs.reserved_jobs(resource_ids_list)
        self.resource_ids_list = resource_ids_list  # 代表这个site拥有的job的id列表

    # 由于资源抢占关系而更新战位的资源列表
    def update_resorces(self, new_resource_ids_list):
        self.resource_ids_list = new_resource_ids_list
