import time
import pandas as pd
import numpy as np

# STOPWATCH_TYPES = [
#     ('Phase', 'i4'),
#     ('Task', 'U'),
#     ('Duration', 'f8'),
#     ('Count', 'i4'),
#     ('Avg', 'f8')
# ]

class Stopwatch:
    def __init__(self):
        # self.laps = pd.DataFrame(columns=['Phase', 'Task', 'Duration', 'Count', 'Avg'], dtype = np.dtype([('Phase', int), ('Task', str), ('Duration', float), ('Count', int), ('Avg', float)])) 
        self.laps = pd.DataFrame(columns=['Phase', 'Task', 'Duration', 'Count', 'Avg']) 

    def start(self):
        self.last_time = time.perf_counter()

    def lap(self, task='', count=0):
        time_now = time.perf_counter()
        task = task.strip()
        idx = len(self.laps)
        self.laps.loc[idx] = [
            idx+1,
            task, 
            time_now - self.last_time,
            count,
            np.nan
            ]
        self.last_time = time.perf_counter()
    
    def lap_by_task(self, task='', count=0):
        time_now = time.perf_counter()
        task = task.strip()
        idx = len(self.laps)
        if self.laps.loc[self.laps['Task']==task].empty:
            self.laps.loc[idx] = [
                idx+1,
                task, 
                time_now - self.last_time,
                count,
                np.nan
                ]
        else:
            self.laps.loc[self.laps['Task']==task, 'Duration'] += np.float64(time_now - self.last_time)
            self.laps.loc[self.laps['Task']==task, 'Count'] += np.int32(count)
        self.last_time = time.perf_counter()
    
    def __calculate_avg(self, row):
        if row['Count'] <= 0:
            return 0
        else:
            return row['Duration']/row['Count']

    def print_result(self):
        self.laps['Avg'] = self.laps.apply(self.__calculate_avg, axis=1)
        print(self.laps.to_string(
            index=False, 
            justify='center',
            formatters={'Duration': lambda x: "{:.9f}".format(x), 'Count': lambda x: str(x) if x != 0 else '', 'Avg': lambda x: "{:.9f}".format(x) if x != 0 else ''}
            ))

if __name__ == '__main__':
    stopwatch = Stopwatch()
    stopwatch.start()
    x = 0
    for i in range(0,1000000):
        x += 1
    stopwatch.lap_by_task('x += 1')
    x = 0
    for i in range(0,2000000):
        x += 1
    stopwatch.lap_by_task('x += 1', 30)
    x = 0
    for i in range(0,200000):
        x += 2
    stopwatch.lap_by_task('x += 2')
    x = 0
    for i in range(0,2000000):
        x += 1
    stopwatch.lap('x += 1')
    stopwatch.print_result()
    