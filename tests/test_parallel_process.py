import concurrent
from concurrent.futures import ProcessPoolExecutor
from time import sleep, time

import numpy as np
import wandb



def slow_task(zero_time ,title: str = 'unnamed task', duration_sec: float = 60) -> None:
    delay = np.random.uniform(0.2, 2)  # [sec]
    loop_counts = int(duration_sec/delay)
    wandb.init(project="parallel test2", entity="yarden92", name=title)
    for i in range(int(loop_counts)):
        print(f'{title} {i}/{loop_counts}')
        t = time() - zero_time
        wandb.log({"progress": i/loop_counts, "t": t})
        # wandb.log({title: i/loop_counts})
        sleep(delay)

    print(f'{title} done')


def parallel_task(function_pointer, num_tasks=7, num_processes: int = 1, duration_min=40, duration_max=60) -> None:
    # loop on n gpus
    zero_time = time()
    print(f'starting {num_tasks} tasks')
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(num_tasks):
            duration = np.random.uniform(duration_min, duration_max)
            executor.submit(function_pointer, zero_time, f'task {i}', duration)

        print('finished loading the tasks.')

    print(f'all tasks done')


if __name__ == '__main__':
    parallel_task(slow_task, num_tasks=10, num_processes=3, duration_min=10, duration_max=20)
