import torch as tc
import utils


class ModelLight(tc.nn.Module):
    '''类似于 pytorch lighting 的 pytorch Module 子类，
    提供一些成员模板，供 trainer 或 tester 使用'''

    def __init__(self):
        super().__init__()
        self.monitor = utils.Monitor()

    def sample_transform(self, sample):
        return sample

    def train_sample_transform(self, sample):
        return self.sample_transform(sample)

    def test_sample_transform(self, sample):
        return self.sample_transform(sample)

    def train_step(self, batch_data):
        raise NotImplementedError

    def on_epoch_end(self):
        pass

    def inference(self, batch_input):
        raise NotImplementedError
