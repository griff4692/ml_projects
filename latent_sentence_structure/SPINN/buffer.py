import torch
from torch.autograd import Variable
from utils import cudify

class Buffer():
    def __init__(self, h_s, c_s, args):
        self.states = list(zip(
            list(torch.split(h_s.squeeze(0), 1, 0)),
            list(torch.split(c_s.squeeze(0), 1, 0))
        ))

        self.args = args

        self.zero_state = (
            cudify(self.args, Variable(torch.zeros(1, self.args.hidden_size), requires_grad=False)),
            cudify(self.args, Variable(torch.zeros(1, self.args.hidden_size), requires_grad=False))
        )

    def pop(self):
        if self.size() == 0:
            raise Exception("Cannot pop from empty buffer")
        top = self.states.pop()
        return (top)

    def peek(self):
        if self.size() == 0:
            return self.zero_state
        return self.states[-1]

    def size(self):
        return len(self.states)
