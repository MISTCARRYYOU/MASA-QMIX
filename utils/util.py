"""
一些通用的工具类的封装
"""
import math

def left_planes(chosen_plane, current_idle_planes):
    res = []
    for eve in current_idle_planes:
        if eve not in chosen_plane:
            res.append(eve)
    return res


def min_but_zero(state_left_time):
    non_zero_list = []
    for eve in state_left_time:
        if eve != 0:
            non_zero_list.append(eve)
    if len(non_zero_list) != 0:
        return min(non_zero_list)
    else:
        return 0


# 将state_left_time中非0的都减去min_time
def advance_by_min_time(min_time, state_left_time):
    res = []
    for eve in state_left_time:
        if eve != 0:
            assert eve >= min_time
            res.append(eve - min_time)
        else:
            res.append(0)
    return res


# 返回飞机在两个战位之间调运的时间
def count_path_on_road(initial_pos, end_pos, speed):
    return math.sqrt((end_pos[0]-initial_pos[0]) ** 2 + (end_pos[1]-initial_pos[1]) ** 2) / speed

