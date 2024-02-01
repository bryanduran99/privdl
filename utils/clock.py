import time
import functools
import torch.distributed as dist

def strftime(seconds=None):
    '''seconds -> %H:%M:%S, will return local time if seconds=None'''
    if seconds is None:
        return time.strftime('%H:%M:%S')
    t = int(seconds)
    H, t = divmod(t, 3600)
    M, S = divmod(t, 60)
    return ':'.join(str(i).zfill(2) for i in [H, M, S])


class Clock:
    '''
    A clock that keeps the seconds since its init
    and the times of next() applied on it.
    state_dict() and load_state_dict() are supported.

    clock = Clock(steps=...)

    clock.time() -> the seconds since its init

    clock.step -> "the times of next() applied on it" - 1

    clock.steps -> total steps of this clock, default None

    next(clock) -> increase clock.step by 1 and return True,
    will return False if clock.step reached clock.steps

    clock.reset() -> reset the clock's step & time
    '''

    def __init__(self, steps=None):
        self.steps = steps
        self.reset()

    def reset(self):
        self.step = -1
        self.time0 = time.time()

    def __next__(self):
        if self.step == self.steps:
            return False # 防止已经达到 steps 的 clock 因再次 next 而爆表
        self.step += 1
        return self.step != self.steps

    def time(self):
        '''the seconds since its init'''
        return time.time() - self.time0

    def __repr__(self):
        msg = ' '.join([
            f'step={self.step}/{self.steps}',
            f'time={strftime(self.time())}'])
        return f'Clock({msg})'

    def state_dict(self):
        return dict(
            steps=self.steps,
            step=self.step,
            time=self.time())

    def load_state_dict(self, Dict):
        self.steps = Dict['steps']
        self.step = Dict['step']
        self.time0 = time.time() - Dict['time']


def interval(seconds):
    '''返回一个装饰器，被装饰的函数每次调用需要至少间隔 seconds 秒，
    否则将直接返回 None。简单起见，被装饰的函数也要求只能返回 None。'''
    def dec(func):
        env = dict(time0=time.time())
        @functools.wraps(func)
        def decfunc(*args, **kwargs):
            if time.time() - env['time0'] < seconds:
                return
            ret = func(*args, **kwargs)
            assert ret is None, '仅允许对返回 None 的函数用 interval 装饰'
            env['time0'] = time.time()
        return decfunc
    return dec


class TrainLoopClock:
    '''为 epoch-batch 循环提供进度计时器 \n
    clock = TrainLoopClock(dataloader, total_epochs) \n
    for batch_data in clock: \n
        clock.epoch -> 当前 epoch\n
        clock.batch -> 当前 batch\n
        clock.time() -> clock 已运行的秒数 \n
        clock.step() -> 当前 step (每个 batch 记为 1 个 step)\n
        clock.progress() -> clock 已读取数据占总数量的比率 \n
        clock.total_time() —> 预估的 clock 总耗时 \n
        print(clock) -> 打印 clock 的运行详情 \n
        clock.log() -> 返回 clock 的 log_dict \n
        clock.check() -> print(clock) & return clock.log() \n
        if clock.epoch_end(): -> 当前 batch 是否为该 epoch 的最后一个 \n
            on_epoch_end()'''

    def __init__(self, dataloader, total_epochs, start_epoch=0):
        assert start_epoch < total_epochs
        self.start_epoch = start_epoch
        self.dataloader = dataloader
        self.total_epochs = total_epochs
        self.batch_steps = len(dataloader)
        self.total_steps = self.batch_steps * total_epochs
        

    def __iter__(self):
        self.time0 = time.time()
        for self.epoch in range(self.start_epoch, self.total_epochs):
            if dist.is_available() and dist.is_initialized():
                time.sleep(0.003)
                self.dataloader.sampler.set_epoch(self.epoch)
            for self.batch, data in enumerate(self.dataloader):
                yield data
                
    def epoch_end(self):
        return self.batch + 1 == self.batch_steps

    def time(self):
        return time.time() - self.time0

    def step(self):
        return (self.epoch - 0) * self.batch_steps + self.batch

    def progress(self):
        return (self.step() + 1) / self.total_steps

    def total_time(self):
        return self.time() / self.progress()

    def __str__(self):
        now = strftime()
        percent = int(self.progress()*100)
        past = strftime(self.time())
        total = strftime(self.total_time())
        epoch = f'{self.epoch}/{self.total_epochs}'
        batch = f'{self.batch}/{self.batch_steps}'
        step = f'{self.step()}/{self.total_steps}'
        return f'{now} Time={percent}%={past}/{total} Epoch={epoch} Batch={batch} Step={step}'

    def log(self):
        return dict(time=self.time(), epoch=self.epoch, batch=self.batch, step=self.step())

    def check(self):
        print(self)
        return self.log()
