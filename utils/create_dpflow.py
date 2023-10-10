import dpflow
import logging
import time
import multiprocessing
import numpy as np


dpflow.pipe.logger.setLevel(logging.ERROR)
dpflow.common.logger.setLevel(logging.ERROR)


def dpflow_worker(worker_id, data_iter, pipe_name,
    seed_funcs=[np.random.seed]):
    '''send obj from data_iter to pipe_name'''
    for seed_func in seed_funcs:
        seed_func(worker_id)
    pipe = dpflow.OutputPipe(pipe_name, buffer_size=16)
    with dpflow.control(io=[pipe]):
        for obj in data_iter:
            pipe.put_pyobj(obj)


def create_dpflow(data_iter, worker_num):
    '''
    yield from data_iter in parallel using dpflow
    
    Remember to .close() the returned generator,
    otherwise the dpflow workers won't stop.
    '''
    pipe_name = f'facerec.xionghuixin.{time.time()}'
    workers = [
        multiprocessing.Process(
            target=dpflow_worker,
            args=(worker_id, data_iter, pipe_name))
        for worker_id in range(worker_num)]
    for worker in workers:
        worker.start()
    pipe = dpflow.InputPipe(pipe_name, buffer_size=16)
    with dpflow.control(io=[pipe]):
        try:
            yield from pipe
        except GeneratorExit:
            for worker in workers:
                worker.terminate()
