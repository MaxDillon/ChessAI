import glob
import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
import maximum.industries.instance_pb2 as instance_pb2

#
# Use np.random for any random numbers drawn in this module. When this module is
# called in worker processes main.py will ensure that each call has a new numpy
# RandomState.
#

DTYPE = 'float32'  # can set this to 'float16', but this breaks batch normalization.
NUM_REAL_PIECES = 8
NUM_INPUT_CHANNELS = NUM_REAL_PIECES * 2 + 1


def piece_to_channel(p, reverse_sides=False):
    if p == 0:
        return 0
    elif p > 127:
        # java bytes are signed, and black pieces are represented as negative numbers.
        # python bytes are unsigned, if jbyte<0 then pbyte>127 and pbyte=256+jbyte.
        return (0 if reverse_sides else NUM_REAL_PIECES) + (256 - p)
    else:
        return p + (NUM_REAL_PIECES if reverse_sides else 0)


def flip(x, y, flip_left_right, reverse_sides):
    xx = 7 - x if flip_left_right else x
    yy = 7 - y if reverse_sides else y
    return xx, yy


def flip_policy_index(index, flip_left_right, reverse_sides):
    src = int(index / 64)
    srcx = src % 8
    srcy = int(src / 8)
    (fsrcx, fsrcy) = flip(srcx, srcy, flip_left_right, reverse_sides)
    fsrc = fsrcy * 8 + fsrcx
    dst = index % 64
    dstx = dst % 8
    dsty = int(dst / 8)
    (fdstx, fdsty) = flip(dstx, dsty, flip_left_right, reverse_sides)
    fdst = fdsty * 8 + fdstx
    return int(fsrc * 64 + fdst)


def entropy(x):
    e = -x * np.log(x)
    return e.sum()


def adjust_entropy(probs):
    metf = 0.4
    max_entropy = -((1 - metf) * np.log((1 - metf) / len(probs)) - metf * np.log(metf))
    probs += 0.001
    probs /= probs.sum()
    max_iter = 15
    while entropy(probs) > max_entropy and max_iter > 0:
        max_iter -= 1
        probs **= 1.5
        probs += 0.001
        probs /= probs.sum()
    return probs


def transform(insts):
    n = len(insts) * 4
    x_input = np.zeros((n, NUM_INPUT_CHANNELS, 8, 8), dtype=DTYPE)
    y_value = np.zeros((n, 1), dtype=DTYPE)
    y_policy = np.zeros((n, 8 * 8 * 8 * 8), dtype=DTYPE)
    for i in range(n):
        flip_left_right = (i & 1) > 0
        reverse_sides = (i & 2) > 0
        inst = insts[int(i/4)]
        board = bytearray(inst.board_state)
        for x in range(8):
            for y in range(8):
                p = int(board[y * 8 + x])
                if p != 0:
                    (xx, yy) = flip(x, y, flip_left_right, reverse_sides)
                    x_input[i, piece_to_channel(p, reverse_sides), yy, xx] = 1
        x_input[i, 0, :, :] = (1 if inst.player == 0 else -1) * (-1 if reverse_sides else 1)
        y_value[i, 0] = inst.outcome * 0.98
        probs = adjust_entropy(np.array([tsr.prob for tsr in inst.tree_search_result]))
        for j in range(len(inst.tree_search_result)):
            tsr = inst.tree_search_result[j]
            y_policy[i, flip_policy_index(tsr.index, flip_left_right, reverse_sides)] = probs[j]
    return x_input, y_value, y_policy


def balance(insts):
    out = []
    total = 0
    counts = [0, 0, 0, 0]
    for inst in insts:
        if inst.outcome != 0:
            which = (1 if inst.player == 0 else 0) + inst.outcome + 1
            prob = 0.30 if (counts[which] + 1.0) / (total + 1.0) > 0.25 else 0.4
            if np.random.uniform() < prob:
                out.append(inst)
                total += 1
                counts[which] += 1
        else:
            if np.random.uniform() < 0.1:
                out.append(inst)
    return out


def load_data(filenames):
    allinsts = []
    for filename in filenames:
        with open(filename, "rb") as f:
            buf = f.read()
            pos = 0
            newinsts = []
            while pos < len(buf):
                msg_len, pos = _DecodeVarint32(buf, pos)
                msg_buf = buf[pos:pos+msg_len]
                pos += msg_len
                inst = instance_pb2.TrainingInstance()
                inst.ParseFromString(msg_buf)
                if len(inst.board_state) == 64:
                    newinsts.append(inst)
                else:
                    print("WTF: bad board length in %s: %d" % (filename, len(inst.board_state)))
            allinsts += newinsts
    return allinsts


def load_balance_transform(pattern, choose_n, from_last_n=0):
    filenames = glob.glob(pattern)
    filenames.sort()
    filenames = filenames[-from_last_n:]
    filenames = [filenames[i] for i in np.random.randint(0, len(filenames), choose_n)]
    return transform(balance(load_data(filenames)))
