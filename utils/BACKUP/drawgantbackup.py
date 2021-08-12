import time
import plotly as py
import plotly.figure_factory as ff

# x轴, 对应于画图位置的起始坐标x
# start, time, of, every, task, , //每个工序的开始时间
n_start_time = [0, 0, 2, 6, 0, 0, 3, 4, 10, 13, 4, 3, 10, 6, 12, 4, 5, 6, 14, 7, 9, 9, 16, 7, 11, 14, 15, 12, 16, 17,
                16, 15, 18, 19, 19, 20, 21, 20, 22, 21, 24, 24, 25, 27, 30, 30, 27, 25, 28, 33, 36, 33, 30, 37, 37, 40]
# length, 对应于每个图形在x轴方向的长度
# duration, time, of, every, task, , //每个工序的持续时间
n_duration_time = [6, 2, 1, 6, 4, 3, 1, 6, 3, 3, 2, 1, 2, 1, 2, 1, 1, 3, 2, 2, 6, 2, 1, 4, 4, 2, 6, 6, 1, 2, 1, 4, 6, 1,
                   6, 1, 1, 1, 5, 6, 1, 6, 4, 3, 6, 1, 6, 3, 2, 6, 1, 4, 6, 1, 5, 6]

# y轴, 对应于画图位置的起始坐标y
# bay, id, of, every, task, , ==工序数目，即在哪一行画线
n_bay_start = [1, 5, 5, 1, 2, 4, 5, 5, 4, 4, 3, 0, 5, 2, 5, 0, 0, 3, 5, 0, 3, 0, 5, 2, 2, 0, 3, 1, 0, 5, 4, 2, 1, 0, 5,
               0, 0, 2, 0, 3, 2, 1, 2, 0, 1, 0, 3, 4, 5, 3, 0, 2, 5, 2, 0, 6]

# 工序号，可以根据工序号选择使用哪一种颜色
# n_job_id = [1, 9, 8, 2, 0, 4, 6, 9, 9, 0, 6, 4, 7, 1, 5, 8, 3, 8, 2, 1, 1, 8, 9, 6, 8, 5, 8, 4, 2, 0, 6, 7, 3, 0, 2, 1, 7, 0, 4, 9, 3, 7, 5, 9, 5, 2, 4, 3, 3, 7, 5, 4, 0, 6, 5]
n_job_id = ['B', 'J', 'I', 'C', 'A', 'E', 'G', 'J', 'J', 'A', 'G', 'E', 'H', 'B', 'F', 'I', 'D', 'I', 'C', 'B', 'B',
            'I', 'J', 'G', 'I', 'F', 'I', 'E', 'C', 'A', 'G', 'H', 'D', 'A', 'C', 'B', 'H', 'A', 'E', 'J', 'D', 'H',
            'F', 'J', 'F', 'C', 'E', 'D', 'D', 'H', 'F', 'E', 'A', 'G', 'F']

op = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

colors = ('rgb(46, 137, 205)',
          'rgb(114, 44, 121)',
          'rgb(198, 47, 105)',
          'rgb(58, 149, 136)',
          'rgb(107, 127, 135)',
          'rgb(46, 180, 50)',
          'rgb(150, 44, 50)',
          'rgb(100, 47, 150)',
          'rgb(58, 100, 180)',
          'rgb(150, 127, 50)')



# belows don't need to be revised, fix it

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
        operation['Task'] = 'M' + str(n_bay_start.__getitem__(index) + 1)
        operation['Start'] = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        operation['Finish'] = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        operation['Resource'] = 'J' + str(job_num)
        df.append(operation)
    df.sort(key=lambda x: x["Task"], reverse=True)
    return df


def draw_prepare():
    df = create_draw_defination()
    return ff.create_gantt(df, colors=colors, index_col='Resource',
                           title='mk01的一个最佳调度', show_colorbar=True,
                           group_tasks=True, data=n_duration_time,
                           showgrid_x=True, showgrid_y=True)


def add_annotations(fig):
    y_pos = 0
    for index in range(len(n_job_id)):
        # 机器，纵坐标
        y_pos = n_bay_start.__getitem__(index)

        x_start = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        print(type(start_time), x_start, n_start_time.__getitem__(index),n_start_time.__getitem__(index) * millis_seconds_per_minutes)

        x_end = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        x_pos = (x_end - x_start) / 2 + x_start

        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        text = 'J(' + str(job_num) + "," + str(get_op_num(job_num)) + ")=" + str(n_duration_time.__getitem__(index))
        # text = 'T' + str(job_num) + str(get_op_num(job_num))
        text_font = dict(size=14, color='black')
        fig['layout']['annotations'] += tuple(
            [dict(x=x_pos, y=y_pos, text=text, textangle=-30, showarrow=False, font=text_font)])


def draw_fjssp_gantt():
    fig = draw_prepare()
    add_annotations(fig)
    py.offline.plot(fig, filename='fjssp-gantt-picture')


if __name__ == '__main__':
    draw_fjssp_gantt()
