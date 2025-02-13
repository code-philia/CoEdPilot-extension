import time
import pandas as pd


class Stopwatch:
    def __init__(self):
        self.laps = pd.DataFrame(columns=['Phase', 'Task', 'Duration'])

    def start(self):
        self.last_time = time.perf_counter()

    def lap(self, task=''):
        time_now = time.perf_counter()
        task = task.strip()
        idx = len(self.laps)
        self.laps.loc[idx] = [
            idx + 1,
            task,
            time_now - self.last_time
        ]
        self.last_time = time.perf_counter()

    def lap_by_task(self, task=''):
        time_now = time.perf_counter()
        task = task.strip()
        idx = len(self.laps)
        if self.laps.loc[self.laps['Task'] == task].empty:
            self.laps.loc[idx] = [
                idx + 1,
                task,
                time_now - self.last_time
            ]
        else:
            self.laps.loc[self.laps['Task'] == task,
                          'Duration'] += time_now - self.last_time
        self.last_time = time.perf_counter()

    def print_result(self):
        print(self.laps.to_string(
            index=False,
            justify='center',
            formatters={'Duration': lambda x: "{:.3f}".format(x)}
        ))


if __name__ == '__main__':
    stopwatch = Stopwatch()
    stopwatch.start()
    x = 0
    for i in range(0, 1000000):
        x += 1
    stopwatch.lap_by_task('x += 1')
    x = 0
    for i in range(0, 2000000):
        x += 1
    stopwatch.lap_by_task('x += 1')
    x = 0
    for i in range(0, 200000):
        x += 2
    stopwatch.lap_by_task('x += 2')
    x = 0
    for i in range(0, 2000000):
        x += 1
    stopwatch.lap('x += 1')
    stopwatch.print_result()
