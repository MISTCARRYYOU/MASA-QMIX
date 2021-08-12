import time
import plotly as py
import plotly.figure_factory as ff
import pickle
job_times = [10, 10, 15, 4, 6, 2, 2, 10, 15]

# with open("../my_data_and_graph/pickles/process_ppo.pk", "rb") as f:
#     all_data = pickle.load(f)
# print(all_data)
# # paras
# # 选择一个过程进行可视化
# one_process = all_data[0]

one_process = \
[(0, 5, 9, 0), (0, 5, 1, 1), (0, 5, 13, 2), (0, 5, 15, 3), (0, 5, 3, 4), (0, 5, 4, 5), (12, 4, 7, 0), (12, 5, 9, 6), (22, 2, 0, 0), (22, 4, 11, 2), (22, 5, 13, 7), (24, 4, 2, 6), (36, 2, 5
, 2), (36, 4, 11, 3), (38, 4, 4, 5), (40, 4, 15, 4), (43, 4, 1, 1), (44, 2, 3, 5), (44, 4, 4, 7), (49, 2, 16, 1), (53, 2, 12, 4), (61, 3, 1, 5), (62, 2, 3, 3), (63, 2, 6, 6), (69, 8, 2, 5)
, (72, 3, 15, 0), (72, 3, 1, 2), (72, 2, 0, 7), (76, 3, 5, 1), (84, 8, 4, 1), (84, 8, 1, 2), (85, 3, 5, 4), (86, 6, 2, 5), (88, 7, 2, 5), (89, 8, 14, 0), (95, 3, 15, 7), (98, 1, 0, 5), (99
, 6, 2, 2), (101, 6, 1, 1), (101, 3, 3, 3), (103, 7, 2, 2), (105, 6, 3, 0), (105, 8, 4, 3), (107, 8, 14, 4), (108, 3, 5, 6), (109, 7, 1, 1), (112, 7, 16, 0), (112, 0, 0, 5), (112, 8, 3, 7)
, (113, 1, 2, 2), (119, 1, 1, 1), (122, 6, 4, 3), (123, 0, 2, 2), (123, 6, 0, 4), (124, 7, 4, 3), (129, 0, 1, 1), (130, 1, 16, 0), (134, 1, 3, 3), (134, 6, 2, 7), (136, 8, 5, 6), (137, 7,
4, 4), (138, 7, 2, 7), (140, 0, 16, 0), (146, 0, 3, 3), (148, 1, 2, 7), (151, 6, 5, 6), (153, 7, 1, 6), (155, 1, 5, 4), (158, 0, 4, 7), (167, 0, 16, 4), (171, 1, 5, 6), (189, 0, 3, 6)]


title = "ppo策略下-" + "船舶调度模型gantt图"


# 将one_process里的数据转化为四个列表
n_start_time = []
n_duration_time = []
n_bay_start = []
n_job_id = []
for eve_tuple in one_process:
    n_start_time.append(int(eve_tuple[0]))
    n_duration_time.append(int(job_times[eve_tuple[1]]))
    n_bay_start.append(int(eve_tuple[2]))
    n_job_id.append(int(eve_tuple[3]))

print(n_start_time,'\n',
n_duration_time,'\n',
n_bay_start,'\n',
n_job_id)
# print(n_bay_start)
# print(all_data)
# x轴, 对应于画图位置的起始坐标x
# start, time, of, every, task, , //每个工序的开始时间
# n_start_time = [0, 0, 2, 6, 0, 0, 3, 4, 10, 13, 4, 3, 10, 6, 12, 4, 5, 6, 14, 7, 9, 9, 16, 7, 11, 14, 15, 12, 16, 17,
#                 16, 15, 18, 19, 19, 20, 21, 20, 22, 21, 24, 24, 25, 27, 30, 30, 27, 25, 28, 33, 36, 33, 30, 37, 37, 40]
# # length, 对应于每个图形在x轴方向的长度
# # duration, time, of, every, task, , //每个工序的持续时间
# n_duration_time = [6, 2, 1, 6, 4, 3, 1, 6, 3, 3, 2, 1, 2, 1, 2, 1, 1, 3, 2, 2, 6, 2, 1, 4, 4, 2, 6, 6, 1, 2, 1, 4, 6, 1,
#                    6, 1, 1, 1, 5, 6, 1, 6, 4, 3, 6, 1, 6, 3, 2, 6, 1, 4, 6, 1, 5, 6]
#
# # y轴, 对应于画图位置的起始坐标y
# # bay, id, of, every, task, , ==工序数目，即在哪一行画线
# n_bay_start = [1, 5, 5, 1, 2, 4, 5, 5, 4, 4, 3, 0, 5, 2, 5, 0, 0, 3, 5, 0, 3, 0, 5, 2, 2, 0, 3, 1, 0, 5, 4, 2, 1, 0, 5,
#                0, 0, 2, 0, 3, 2, 1, 2, 0, 1, 0, 3, 4, 5, 3, 0, 2, 5, 2, 0, 6]
#
# # 工序号，可以根据工序号选择使用哪一种颜色
# # n_job_id = [1, 9, 8, 2, 0, 4, 6, 9, 9, 0, 6, 4, 7, 1, 5, 8, 3, 8, 2, 1, 1, 8, 9, 6, 8, 5, 8, 4, 2, 0, 6, 7, 3, 0, 2, 1, 7, 0, 4, 9, 3, 7, 5, 9, 5, 2, 4, 3, 3, 7, 5, 4, 0, 6, 5]
# n_job_id = ['B', 'J', 'I', 'C', 1, 'E', 'G', 'J', 'J', 1, 'G', 'E', 'H', 'B', 'F', 'I', 'D', 'I', 'C', 'B', 'B',
#             'I', 'J', 'G', 'I', 'F', 'I', 'E', 'C', 1, 'G', 'H', 'D', 1, 'C', 'B', 'H', 1, 'E', 'J', 'D', 'H',
#             'F', 'J', 'F', 'C', 'E', 'D', 'D', 'H', 'F', 'E', 1, 'G', 'F', "F"]

print(len(n_bay_start), len(n_job_id))

# belows are the number of planes
op = [0, 1, 2, 3, 4, 5, 6, 7]
colors = ('rgb(46, 137, 205)',
          'rgb(114, 44, 121)',
          'rgb(198, 47, 105)',
          'rgb(58, 149, 136)',
          'rgb(107, 127, 135)',
          'rgb(46, 180, 50)',
          'rgb(150, 44, 50)',
          'rgb(100, 47, 150)')

millis_seconds_per_minutes = 1000 * 60
start_time = time.time() * 1000

job_sumary = {}


# 获取工件对应的第几道工序
def get_op_num(job_num):
    index = job_sumary.get(str(job_num))
    new_index = 1
    if index:
        new_index = index + 1
    job_sumary[str(job_num)] = new_index
    return new_index


def create_draw_defination():
    df = []
    for index in range(len(n_job_id)):
        operation = {}
        # 机器，纵坐标
        operation['Task'] = 'Site' + str(n_bay_start.__getitem__(index) + 1)
        operation['Start'] = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        operation['Finish'] = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        operation['Resource'] = 'P' + str(job_num)
        df.append(operation)
        # print(operation['Task'])
    df.sort(key=lambda x: int(x["Task"][4:]), reverse=True)
    return df


def draw_prepare():
    df = create_draw_defination()
    return ff.create_gantt(df, colors=colors, index_col='Resource',
                           title=title, show_colorbar=True,
                           group_tasks=True, data=n_duration_time,
                           showgrid_x=False, showgrid_y=True)


def add_annotations(fig):
    y_pos = 0
    for index in range(len(n_job_id)):
        # 机器，纵坐标
        y_pos = n_bay_start.__getitem__(index)

        x_start = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        # a= start_time.__add__(16)
        # print(type(start_time), x_start, type(n_start_time.__getitem__(index)),index,a)
        x_end = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        x_pos = (x_end - x_start) / 2 + x_start

        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        text = 'P(' + str(job_num) + "," + str(get_op_num(job_num)) + ")=" + str(n_duration_time.__getitem__(index))
        # text = 'T' + str(job_num) + str(get_op_num(job_num))
        text_font = dict(size=14, color='black')
        # print(x_pos, y_pos, text)
        fig['layout']['annotations'] += tuple(
            [dict(x=x_pos, y=y_pos+0.2, text=text, textangle=-15, showarrow=False, font=text_font)])


def draw_fjssp_gantt():
    fig = draw_prepare()
    add_annotations(fig)
    py.offline.plot(fig, filename='boat-gantt-picture.html')


if __name__ == '__main__':
    draw_fjssp_gantt()
