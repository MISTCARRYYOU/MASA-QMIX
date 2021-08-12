"""
这个文件用于描述task的内容

"""
from utils.job import Jobs

class Task:
    def __init__(self):
        # 当前仅考虑最简单的任务，串行任务
        self.simple_task = [5, 4, 2, 3, 8, 6, 7, 1, 0]  # 时间最快是74完成
        jobs = Jobs()
        # 带有对象的
        self.simple_task_object = []
        for eve in self.simple_task:
            self.simple_task_object.append(jobs.jobs_object_list[eve])


