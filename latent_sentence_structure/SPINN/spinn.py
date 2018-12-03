import torch
import torch.nn as nn
from torch.autograd import Variable
from actions import Reduce
from constants import PAD, SHIFT, REDUCE
from buffer import Buffer
from stack import create_stack
from tracking_lstm import TrackingLSTM
from random import random
from utils import cudify
import math

class SPINN(nn.Module):
    def __init__(self, args):
        super(SPINN, self).__init__()
        self.args = args

        self.dropout = nn.Dropout(p=self.args.dropout_rate_input)
        self.batch_norm1 = nn.BatchNorm1d(self.args.hidden_size * 2)

        self.word = nn.Linear(self.args.embed_dim, self.args.hidden_size * 2)
        self.reduce = Reduce(self.args)

        self.track = None
        if self.args.tracking:
            self.track = TrackingLSTM(self.args)

    def update_tracker(self, buffer, stack, batch_size):
        b_s, s1_s, s2_s = [], [], []
        for b_id in range(batch_size):
            b = buffer[b_id].peek()[0]
            s1, s2 = stack[b_id].peek_two()
            b_s.append(b); s1_s.append(s1[0]); s2_s.append(s2[0])

        bs_cat = torch.cat(b_s)
        s1s_cat = torch.cat(s1_s)
        s2s_cat = torch.cat(s2_s)

        tracking_inputs = torch.cat([bs_cat, s1s_cat, s2s_cat], dim=1)
        return self.track(tracking_inputs)

    def resolve_action(self, buffer, stack, buffer_size, stack_size, act, time_stamp, ops_left):
        # must pad
        if buffer_size == 0 and stack_size == 1:
            raise Exception("Should have immediately returned PAD inside main loop.")

        if buffer_size == 0:
            return REDUCE, True

        if stack_size < 2:
            return SHIFT, True

        # must reduce
        if stack_size >= ops_left:
            return REDUCE, True

        # must shift
        if stack_size < 2:
            return SHIFT, True

        return act, False

    def forward(self, sentence, transitions, num_ops, other_sent, teacher_prob):
        batch_size, sent_len, _  = sentence.size()
        out = self.word(sentence) # batch, |sent|, h * 2s

        # batch normalization and dropout
        if not self.args.no_batch_norm:
            out = out.transpose(1, 2).contiguous()
            out = self.batch_norm1(out) # batch,  h * 2, |sent| (Normalizes batch * |sent| slices for each feature
            out = out.transpose(1, 2)

        if self.args.dropout_rate_input > 0:
            out = self.dropout(out) # batch, |sent|, h * 2

        (h_sent, c_sent) = torch.chunk(out, 2, 2)  # ((batch, |sent|, h), (batch, |sent|, h))

        buffer_batch = [Buffer(h_s, c_s, self.args) for h_s, c_s
            in zip(
                list(torch.split(h_sent, 1, 0)),
                list(torch.split(c_sent, 1, 0))
            )
        ]

        stack_batch = [
            create_stack(self.args)
            for _ in buffer_batch
        ]

        if self.args.tracking:
            self.track.initialize_states(other_sent)
        else:
            assert transitions is not None

        if transitions is None:
            num_transitions = (2 * sent_len) - 1
        else:
            transitions_batch = [trans.squeeze(1) for trans
                in list(torch.split(transitions, 1, 1))]
            num_transitions = len(transitions_batch)

        lstm_actions, true_actions = [], []

        for time_stamp in range(num_transitions):
            ops_left = num_transitions - time_stamp

            reduce_ids = []
            reduce_lh, reduce_lc = [], []
            reduce_rh, reduce_rc = [], []
            reduce_valences = []
            reduce_tracking_states = []
            teacher_valences = None
            if self.args.tracking:
                valences, tracking_state = self.update_tracker(buffer_batch, stack_batch, batch_size)
                _, pred_trans = valences.max(dim=1)
                if self.training and self.args.teacher:
                    use_teacher = True # TODO for now always use teacher - later --> random() < teacher_prob
                    if use_teacher and self.args.continuous_stack:
                        teacher_valences = cudify(self.args, Variable(torch.zeros(valences.size()), requires_grad=False))

                    temp_trans = transitions_batch[time_stamp]

                    for b_id in range(batch_size):
                        if temp_trans[b_id].data[0] > PAD:
                            true_actions.append(temp_trans[b_id])
                            lstm_actions.append(valences[b_id].unsqueeze(0))

                            if teacher_valences is not None:
                                teacher_valences[b_id, temp_trans[b_id].data[0]] = 1.0

                    temp_trans = temp_trans.data if use_teacher else pred_trans.data
                else:
                    temp_trans = pred_trans.data
            else:
                valences = None
                temp_trans = transitions_batch[time_stamp].data

            for b_id in range(batch_size):
                stack_size, buffer_size = stack_batch[b_id].size(), buffer_batch[b_id].size()
                # this sentence is done!
                my_ops_left = num_ops[b_id] - time_stamp
                if my_ops_left <= 0:
                    # should coincide with teacher padding or else num_ops has a bug
                    if self.training and self.args.teacher:
                        assert temp_trans[b_id] == PAD
                    continue
                else:
                    act = temp_trans[b_id]

                    # ensures it's a valid act according to state of buffer, batch, and timestamp
                    # safe check actions if not using teacher forcing... or using teacher forcing but in evaluation
                    if self.args.tracking and (not self.args.teacher or (self.args.teacher and not self.training)):
                        act, act_ignored = self.resolve_action(buffer_batch[b_id],
                            stack_batch[b_id], buffer_size, stack_size, act, time_stamp, my_ops_left)

                if self.args.tracking:
                    # use teacher valences over predicted valences
                    if teacher_valences is not None:
                        reduce_valence, shift_valence = teacher_valences[b_id]
                    else:
                        reduce_valence, shift_valence = valences[b_id]
                else:
                    reduce_valence, shift_valence = None, None

                no_action = True

                # 2 - REDUCE
                if act == REDUCE or (self.args.continuous_stack and not self.args.teacher and stack_size >= 2):
                    no_action = False
                    reduce_ids.append(b_id)

                    r = stack_batch[b_id].peek()
                    if not stack_batch[b_id].pop(reduce_valence):
                        print(sentence[b_id, :, :].sum(dim=1), transitions[b_id, :])
                        raise Exception("Tried to pop from an empty list.")

                    l = stack_batch[b_id].peek()
                    if not stack_batch[b_id].pop(reduce_valence):
                        print(sentence[b_id, :, :].sum(dim=1), transitions[b_id, :])
                        raise Exception("Tried to pop from an empty list.")

                    reduce_lh.append(l[0]); reduce_lc.append(l[1])
                    reduce_rh.append(r[0]); reduce_rc.append(r[1])

                    if self.args.tracking:
                        reduce_valences.append(reduce_valence)
                        reduce_tracking_states.append(tracking_state[b_id].unsqueeze(0))

                # 3 - SHIFT
                if act == SHIFT or (self.args.continuous_stack and not self.args.teacher and buffer_size > 0):
                    no_action = False
                    word = buffer_batch[b_id].pop()
                    stack_batch[b_id].add(word, shift_valence, time_stamp)

                if no_action:
                    print("\n\nWarning: Didn't choose an action.  Look for a bug!  Attempted %d action but was denied!" % act)

            if len(reduce_ids) > 0:
                h_lefts = torch.cat(reduce_lh)
                c_lefts = torch.cat(reduce_lc)
                h_rights = torch.cat(reduce_rh)
                c_rights = torch.cat(reduce_rc)

                if self.args.tracking:
                    e_out = torch.cat(reduce_tracking_states)
                    h_outs, c_outs = self.reduce((h_lefts, c_lefts), (h_rights, c_rights), e_out)
                else:
                    h_outs, c_outs = self.reduce((h_lefts, c_lefts), (h_rights, c_rights))

                for i, state in enumerate(zip(h_outs, c_outs)):
                    reduce_valence = reduce_valences[i] if self.args.tracking else None
                    stack_batch[reduce_ids[i]].add(state, reduce_valence)

        outputs = []
        for (i, stack) in enumerate(stack_batch):
            if not self.args.continuous_stack:
                if not stack.size() == 1:
                    print("Stack size is %d.  Should be 1" % stack.size())
                    assert stack.size() == 1
            top_h = stack.peek()[0]
            outputs.append(top_h)

        if len(true_actions) > 0 and self.training:
            return torch.cat(outputs), torch.cat(true_actions), torch.log(torch.cat(lstm_actions))
        return torch.cat(outputs), None, None
