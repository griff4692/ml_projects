import torch
from torch.autograd import Variable
import numpy as np
from random import random
import abc, six
from abc import ABCMeta
import math
from utils import cudify

def create_stack(args):
    if args.continuous_stack:
        return ContinuousStack(args)
    else:
        return DefaultStack(args)

@six.add_metaclass(ABCMeta)
class BaseStack:
    @abc.abstractmethod
    def add(self, state, valence, id=0):
        pass

    @abc.abstractmethod
    def pop(self, valence):
        pass

    @abc.abstractmethod
    def peek(self):
        pass

    @abc.abstractmethod
    def peek_two(self):
        pass

class DefaultStack(BaseStack):
    # TODO Figure out Thin Stack.
    def __init__(self, args):
        self.args = args
        self.states = []
        self.dim = args.hidden_size
        self.zero_state = (cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)),
                cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)))

    def add(self, state, valence, id=0):
        self.states.append(state)

    def pop(self, valence):
        try:
            self.states.pop()
            return True
        except IndexError:
            return False

    def peek(self):
        if self.size() == 0:
            return self.zero_state

        top = self.states[-1]
        return top

    def peek_two(self):
        if self.size() == 0:
            return self.zero_state, self.zero_state
        if self.size() == 1:
            return self.states[-1], self.zero_state

        return self.states[-1], self.states[-2]

    def size(self):
        return len(self.states)

class ContinuousStack(BaseStack):
    def __init__(self, args):
        self.args = args
        self.dim = self.args.hidden_size
        self.valences = None
        self.hs = None
        self.cs = None
        self.num_push = 0
        self.num_pop = 0

        self.zero_state = (cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)),
                cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)))


    def one_valence(self):
        return cudify(self.args, Variable(torch.FloatTensor([1]), requires_grad=False))

    def add(self, state, valence, id=0):
        assert len(state) == 2

        self.num_push += 1

        hs, cs = state

        # TODO this is defensive programming but may not be necessary
        valence = valence.clone()

        if self.valences is None:
            self.valences = valence
            self.hs, self.cs = hs, cs
        else:
            if not valence.size()[0] == 1:
                raise Exception("Adding more than one valence at a time.")

            self.valences = torch.cat([self.valences, valence], 0)
            self.hs = torch.cat([self.hs, hs], 0)
            self.cs = torch.cat([self.cs, cs], 0)


    def reduce(self, mass_remaining):
        mass_remaining = cudify(self.args, Variable(torch.FloatTensor([mass_remaining])))
        size = self.size()
        read_mask = cudify(self.args, Variable(torch.zeros(size, 1), requires_grad=False))
        idx = size - 1
        while mass_remaining.data[0] > 0.0 and idx >= 0:
            mass_remaining_data = mass_remaining.data[0]
            this_valence = self.valences[idx].data[0]
            if mass_remaining_data - this_valence >= 1.0:
                mass_coeff = self.valences[idx]
            elif mass_remaining_data > 1.0 and mass_remaining_data - this_valence < 1.0:
                skip_mass = mass_remaining - 1.0
                mass_coeff = self.valences[idx] - skip_mass
                read_mask[idx] = mass_coeff
            else:
                mass_coeff = torch.min(torch.cat([self.valences[idx], mass_remaining]))
                read_mask[idx] = mass_coeff

            mass_remaining -= mass_coeff
            idx -= 1

        reduced_hs = torch.mul(read_mask, self.hs).sum(0, keepdim=True)
        reduced_cs = torch.mul(read_mask, self.cs).sum(0, keepdim=True)
        return reduced_hs, reduced_cs

    def peek(self):
        if self.size() == 0:
            return self.zero_state
        return self.reduce(1.0)

    def peek_two(self):
        if self.size() == 0:
            peek1 = self.zero_state
            peek2 = self.zero_state
        else:
            peek1 = self.reduce(1.0)
            peek2 = self.reduce(2.0)

        return peek1, peek2

    def size(self):
        if self.valences is None:
            return 0

        val_sum = self.valences.sum().data
        if self.args.gpu > -1:
            val_sum = val_sum.cpu()
        if val_sum.numpy()[0] == 0.0:
            return 0

        return self.valences.size()[0]

    def pop(self, valence):
        self.num_pop += 1
        size = self.size()
        idx = size - 1
        mass_remaining = valence.clone()
        while mass_remaining.data[0] > 0.0 and idx >= 0:
            mass_coeff = torch.min(torch.cat([self.valences[idx], mass_remaining]))
            self.valences[idx] = self.valences[idx] - mass_coeff
            mass_remaining -= mass_coeff
            idx -= 1
        return True

    def restore(self, valence):
        self.reduce('restore', valence)


# register all subclasses to base class
BaseStack.register(DefaultStack)
BaseStack.register(ContinuousStack)

if __name__=='__main__':
    def rand_vec(dim):
        return np.random.rand(dim,)

    dim = 1
    s = Stack(dim)

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print("Read is %.2f" % s.peek()[0])

    print("Popping 0.8")
    s.pop(0.8)
    print("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.9))
    s.add(vec, 0.9)
    print("Read is %.2f" % s.peek()[0])

    print("Popping 0.5")
    s.pop(0.5)
    print("Read is %.2f" % s.peek()[0])
