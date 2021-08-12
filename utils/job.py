"""
这个文件书写作业类，即抽象保障资源类

"""


class Jobs:
    def __init__(self):
        # 创建所有的job对象，并用列表index作为id的索引，相当于字典
        self.jobs_object_list = []
        # id: 0 1 2 3 4 5 6 7 8
        jobs_codes = ["ZCTF", "SBTF", "JY", "TYY", "TD", "YQ", "DQ", "GDDE", "GD"]
        jobs_names = ["座舱", "设备舱", "加油", "液压", "供电", "氧气", "氮气", "惯导", "挂弹"]
        jobs_times = [10, 10, 15, 4, 6, 2, 2, 10, 15]
        for i in range(len(jobs_names)):
            temp_object = Job(i, jobs_codes[i], jobs_times[i], jobs_times[i])
            self.jobs_object_list.append(temp_object)

    # 保留的jobs有哪些，因为不同保障位置的jobs不同
    # reserved_job_id :[0,2,3,1,...]
    def reserved_jobs(self, reserved_job_id):
        assert type(reserved_job_id[0]) == int
        reserved_jobs = []
        for id in reserved_job_id:
            reserved_jobs.append(self.jobs_object_list[id])
        self.jobs_object_list = reserved_jobs


class Job:
    def __init__(self, index_id, codes, name, time_span):
        self.index_id = index_id
        self.codes = codes
        self.name = name
        self.time_span = time_span
