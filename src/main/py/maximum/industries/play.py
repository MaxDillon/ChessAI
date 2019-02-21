import chess
import chess.pgn
import getopt
import sys
import scipy.special
import numpy as np
import tensorflow as tf
from ast import literal_eval
from tensorflow.keras import backend as K


def to_model_input(board):
    # 8x8 array of channel IDs
    x = np.array([0 if p is None else p.piece_type + (0 if p.color else 8)
                  for p in [board.piece_at(i) for i in range(64)]]).reshape(8, 8)
    # convert to unmoved kings and rooks based on castling rights
    if board.castling_rights & chess.BB_A1:
        x[0, 4] = 8
        x[0, 0] = 7
    if board.castling_rights & chess.BB_H1:
        x[0, 4] = 8
        x[0, 7] = 7
    if board.castling_rights & chess.BB_A8:
        x[7, 4] = 16
        x[7, 0] = 15
    if board.castling_rights & chess.BB_H8:
        x[7, 4] = 16
        x[7, 7] = 15
    # broadcast convert to 17x8x8 input
    y = 1 * (np.arange(17).reshape((17, 1, 1)) == x)
    # augment by including all 4 reflections
    z = np.stack((y, np.flip(y, axis=2), np.flip(y, axis=1), np.flip(np.flip(y, axis=1), axis=2)))
    z[2:4, 1:17] = z[2:4, [9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]]
    z[0:2, 0, :, :] = 1 if board.turn else -1
    z[2:4, 0, :, :] = -1 if board.turn else 1
    return z


def policy_index(move, rotation):
    uci = move.uci()
    r1, c1 = int(uci[1]) - 1, ord(uci[0]) - ord('a')
    r2, c2 = int(uci[3]) - 1, ord(uci[2]) - ord('a')
    if rotation % 2 > 0:
        c1, c2 = 7 - c1, 7 - c2
    if rotation > 1:
        r1, r2 = 7 - r1, 7 - r2
    return 64 * (8 * r1 + c1) + (8 * r2 + c2)


def score_to_odds(score):
    prob = (0.9999 * score + 1.0) / 2.0
    return prob / (1.0 - prob)


def odds_to_score(odds):
    prob = odds / (odds + 1.0)
    return prob * 2.0 - 1.0


class Engine(object):
    
    def __init__(self, model_path, args, quiet=False):
        # Rather than using load_model('model.h5') to get a keras model, we'll load
        # the same frozen model we use on the java side since this runs faster. Keras
        # complains when we try to construct a Model from the input and output tensors
        # so we'll use the lower level tensorflow session.run API.
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        _ = tf.saved_model.loader.load(self.session,
                                       [tf.saved_model.tag_constants.SERVING],
                                       model_path)
        self.input = self.graph.get_tensor_by_name('input:0')
        self.outputs = [self.graph.get_tensor_by_name('value/Tanh:0'),
                        self.graph.get_tensor_by_name('policy/Softmax:0')]
        
        self.iterations = int(args['iter']) if 'iter' in args else 200
        self.exploration = float(args['expl']) if 'expl' in args else 0.3
        self.temperature = float(args['temp']) if 'temp' in args else 0.1
        self.ramp = int(args['ramp']) if 'ramp' in args else 10
        self.priority_uniform = float(args['unif']) if 'unif' in args else 1.0
        self.priority_exponent = float(args['pexp']) if 'pexp' in args else 2.0
        self.value_in_log_odds = float(args['vilo']) if 'vilo' in args else 0.0
        self.parent_prior_odds_mult = float(args['ppom']) if 'ppom' in args else 0.0
        self.parent_prior_value_diff = float(args['ppvd']) if 'ppvd' in args else 0.0
        self.backprop_win_loss = int(args['bpwl']) if 'bpwl' in args else 0
        self.policy_add_value_and_prior = int(args['pavp']) if 'pavp' in args else 0
        self.take_or_avoid_knowns = int(args['toak']) if 'toak' in args else 0
        self.move_choice_value_quantile = float(args['mcvq']) if 'mcvq' in args else 0

        self.board = chess.Board(fen=chess.STARTING_FEN)
        self.root = [None, 0, 0.0, 0.0]
        self.past_fens = set()
        self.check_repetition = False
        self.quiet = quiet

    # start a new game, optionally from a given position
    def start(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen=fen)
        self.root = [None, 0, 0.0, 0.0]
        self.past_fens = set()
        self.check_repetition = False

    def position(self, toks):
        fen = chess.STARTING_FEN
        pos = 1
        if toks[0] == 'fen':
            fen = ' '.join(toks[1:7])
            pos = 7
        if len(toks) > pos and toks[pos] == 'moves':
            pos += 1
        sync = chess.Board(fen=fen)
        if len(toks) > pos:
            for uci in toks[pos:]:
                sync.push(chess.Move.from_uci(uci))
        target_fen = sync.fen()
        if target_fen == self.board.fen():
            return
        for move in self.board.legal_moves:
            self.board.push(move)
            next_fen = self.board.fen()
            self.board.pop()
            if target_fen == next_fen:
                self.make_move(move)
                return
        self.start(target_fen)
        
    # search for the best move to make from the current position
    def search(self):
        for _ in range(self.iterations):
            state = self.board.copy()
            stack = []
            node = self.root
            while node[0] is not None and not state.is_game_over():
                stack.append(node)
                # Node is previously expanded so we've already computed legal moves.
                moves = [m for m in node[0].keys()]
                priorities = [self.priority(node, move) for move in moves]
                move = moves[np.argmax(priorities)]
                state.push(move)
                node = node[0][move]
            self.expand(state, node)
            self.backprop(stack, node)
        if self.move_choice_value_quantile > 0:
            move = self.pick_move_by_value()
        else:
            move = self.pick_move_by_count()
        self.make_move(move)
        return move

    # make a chosen move
    def make_move(self, move):
        if self.board.is_irreversible(move):
            self.past_fens = set()
            self.check_repetition = False

        self.board.push(move)

        short_fen = ' '.join(self.board.fen().split(' ')[0:2])
        if short_fen in self.past_fens:
            self.check_repetition = True
        self.past_fens.add(short_fen)

        if self.root[0] is not None:
            self.root = self.root[0][move]
        else:
            self.root = [None, 0, 0.0, 0.0]

    # the value of the current state for the given player
    def value(self, for_white):
        self_value = self.root[2] / max(1, self.root[1])
        white_turn = self.board.turn
        sign = 1 if white_turn == for_white else -1
        return self_value * sign

    def priority(self, node, move):
        child_node = node[0][move]
        if child_node[1] > 0:
            # estimated value of the child position from the perspective of the parent.
            move_value = -child_node[2] / child_node[1]
        else:
            # if we haven't evaluated child yet, estimate its prior value (from the perspective of
            # the parent) as a little worse than the value of the parent. In principle the best move
            # should have a value about equal to or a little better than the parent. We do this
            # instead of letting the prior be zero because when we are disadvantaged we don't want
            # unevaluated low probability moves to start with higher move_value than evaluated high
            # probability moves, as this would force too high a branching factor (and conversely
            # too low a branching factor when advantaged).
            if self.parent_prior_odds_mult > 0:
                move_value = odds_to_score(score_to_odds(node[2] / node[1]) * self.parent_prior_odds_mult)
            elif self.parent_prior_value_diff > 0:
                move_value = score_to_odds(node[2] / node[1]) - self.parent_prior_value_diff
            else:
                move_value = np.random.random() * 0.0001
        if self.value_in_log_odds > 0:
            multiplier = min(0.999, max(0.001, self.value_in_log_odds))
            # in the limit as the vilo multiplier approaches zero this transformation has no effect.
            # conversely, as it approaches one it has very strong effect with move_value going to
            # infinity. The multiplier must be capped to avoid this, and it should probably be in
            # the vicinity of 0.9.
            move_value = np.log(score_to_odds(move_value * multiplier)) / 2.0 / multiplier
        info_value = (self.exploration / 2.0 *
                      (((node[1] ** 0.5) / (1 + child_node[1])) ** self.priority_exponent) *
                      (self.priority_uniform / len(node[0]) + child_node[3]))
        return move_value + info_value
        
    def expand(self, state, node):
        if state.is_game_over():
            result = state.result()
            if result == '1-0':
                value = 1.0 if state.turn else -1.0
            elif result == '0-1':
                value = -1.0 if state.turn else 1.0
            else:
                value = 0.0
            node[1] += 1
            node[2] = node[1] * value
        else:
            value, policy = self.session.run(self.outputs, feed_dict={self.input: to_model_input(state)})
            node[0] = {}
            node[1] = 1
            node[2] = value.mean()
            if node[2] < 0 and self.check_repetition and state.can_claim_threefold_repetition():
                # we are expanding a node with losing self-value, so the parent might be tempted
                # to choose it. But if it does the opponent can claim a draw, so the true value
                # of the node is zero.  TODO: check logic
                node[2] = 0.0
            for m in state.legal_moves:
                m_prior = np.mean([policy[i, policy_index(m, i)] for i in range(4)])
                node[0][m] = [None, 0, 0.0, m_prior]

    def backprop(self, stack, node):
        val = node[2] / node[1]
        if self.backprop_win_loss and val == 1.0:
            self.backprop_win(stack, len(stack))
        elif self.backprop_win_loss and val == -1.0:
            self.backprop_loss(stack, len(stack))
        else:
            self.backprop_norm(stack, len(stack), val)
            
    def backprop_norm(self, stack, result_level, val):
        for i in range(result_level, 0, -1):
            j = i - 1
            if stack[j][2] == stack[j][1]:
                # if we were searching non-winning moves below an already known won state,
                # treat it during backprop like it's won.
                return self.backprop_loss(stack, j + 1)
            if stack[j][2] == 0.0 and stack[j][1] == 1:
                # this only happens if the node is a loser with choice of a draw. In this
                # case switch to backpropping zero.
                val = 0.0
            stack[j][2] += val * ((-1) ** (result_level - j))
            stack[j][1] += 1

    def backprop_loss(self, stack, loss_level):
        parent = loss_level - 1
        stack[parent][1] += 1
        stack[parent][2] = stack[parent][1]
        if parent > 0:
            self.backprop_win(stack, parent)

    def backprop_win(self, stack, win_level):
        parent = win_level - 1
        stack[parent][1] += 1
        all_won = True
        for move, node in stack[parent][0].items():
            if node[1] == 0 or node[1] != node[2]:
                all_won = False
                break
        if all_won:
            stack[parent][2] = -stack[parent][1]
            if parent > 0:
                self.backprop_loss(stack, parent)
        else:
            stack[parent][2] -= 1.0
            if parent > 0:
                self.backprop_norm(stack, parent, -1.0)

    def effective_temperature(self):
        n = int(len(self.board.move_stack) / 2)
        p = max(0.0, min(1.0, n / self.ramp))
        return p * self.temperature + (1.0 - p) * min(1.0, self.temperature * 10)
        
    def pick_move_by_count(self):
        nodes = self.root[0]
        moves = [move for move in nodes.keys()]
        counts = np.array([nodes[move][1] for move in moves])
        values = np.array([-nodes[move][2] for move in moves]) / np.maximum(1, counts)
        priors = np.array([nodes[move][3] for move in moves])
        evals = counts + self.policy_add_value_and_prior * (values + priors)
        if self.take_or_avoid_knowns:
            evals += 10000 * (values == 1.0)
            evals -= 10000 * (values == -1.0)
            evals = np.maximum(0.000001, evals)
        evals = evals / evals.sum()
        evals = evals ** (1 / self.effective_temperature())
        evals = evals / evals.sum()
        if not self.quiet:
            priorities = [self.priority(self.root, move) for move in moves]
            for i in range(len(moves)):
                print('%s: %5.3f  (%4d %7.3f %6.3f) %8.5f' % (moves[i].uci(), evals[i],
                                                              nodes[moves[i]][1],
                                                              nodes[moves[i]][2]/max(1, nodes[moves[i]][1]),
                                                              nodes[moves[i]][3], priorities[i]))
        which = np.random.choice(len(moves), p=evals)
        return moves[which]

    def pick_move_by_value(self):
        nodes = self.root[0]
        moves = [move for move in nodes.keys()]
        counts = np.array([nodes[move][1] for move in moves])
        values = np.array([-nodes[move][2] for move in moves]) / np.maximum(1, counts)
        # convert score to probability:  0 < p < 1
        probs = (values * 0.9999 + 1.0) / 2.0
        # for known values force very high precision
        if self.take_or_avoid_knowns:
            counts += (np.abs(values) == 1.0) * 10000
        # add small constants to ensure nonzero alpha,beta when count is zero, and to create
        # a negative bias when counts are low.
        alpha = probs * (counts + 0.001)
        beta = (1 - probs) * (counts + 1.0)
        # base choice primarily on quantile of beta distribution
        quantiles = scipy.special.betaincinv(alpha, beta, self.move_choice_value_quantile)
        # add a random component based on temperature
        randoms = np.random.beta(alpha, beta) * self.temperature
        if not self.quiet:
            priorities = [self.priority(self.root, move) for move in moves]
            for i in range(len(moves)):
                print('%s: %5.3f  (%4d %7.3f %6.3f) %8.5f' % (moves[i].uci(), quantiles[i],
                                                              nodes[moves[i]][1],
                                                              nodes[moves[i]][2]/max(1, nodes[moves[i]][1]),
                                                              nodes[moves[i]][3], priorities[i]))
        which = np.argmax(quantiles + randoms)
        return moves[which]


def argdict(argstr):
    dictfmt = ['"%s": %s' % tuple(argval.split('=')) for argval in argstr.split(",")]
    return literal_eval('{ %s }' % ','.join(dictfmt))


def uci_engine_loop(engine):

    def uci(_):
        print('id name yace')
        print('uciok')

    def isready(_):
        print('readyok')

    def ucinewgame(_):
        engine.start(chess.STARTING_FEN)

    def position(toks):
        engine.position(toks)

    def go(_):
        print('bestmove %s' % engine.search().uci())

    def quit(_):
        sys.exit(0)

    while True:
        line = input()
        tokens = line.split(' ')
        command = tokens[0]
        if command in locals():
            locals()[command](tokens[1:])


def main(argv):
    opts, _ = getopt.getopt(argv, '', ['fen=', 'uci', 'quiet=',
                                       'white=', 'wargs=',
                                       'black=', 'bargs=',
                                       'model=', 'uargs='])
    opts = dict(opts)
    uci = '--uci' in opts
    fen = opts['--fen'] if '--fen' in opts else chess.STARTING_FEN
    turn = 0 if 'w' in fen else 1
    wmodel = opts['--white'] if '--white' in opts else ''
    bmodel = opts['--black'] if '--black' in opts else ''
    umodel = opts['--model'] if '--model' in opts else ''
    wargs = argdict(opts['--wargs']) if '--wargs' in opts else {}
    bargs = argdict(opts['--bargs']) if '--bargs' in opts else {}
    uargs = argdict(opts['--uargs']) if '--uargs' in opts else {}
    quiet = opts['--quiet'].lower() == 'true' if '--quiet' in opts else uci

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    if uci:
        uci_engine_loop(Engine(umodel, uargs, quiet=quiet))

    print('%s\nfen:   %s\nwhite: %s\nblack: %s' % ('#' * 40, fen, wmodel, bmodel))
    print('wargs: %s\nbargs: %s\n%s\n' % (wargs, bargs, '#' * 40))
    
    white = Engine(wmodel, wargs, quiet=quiet)
    black = Engine(bmodel, bargs, quiet=quiet)

    while True:
        white.start(fen)
        black.start(fen)
        engines = [white, black]

        board = white.board
        print(board)
        
        while not board.is_game_over():
            print("Turn: %s" % ['White', 'Black'][turn])
            move = engines[turn].search()
            engines[1 - turn].make_move(move)
            turn = 1 - turn

            print("%s: %6.3f %6.3f" % (move, white.value(True), black.value(False)))
            print(board)
            print(board.fen())

            if '--fen' not in opts:
                game = chess.pgn.Game().without_tag_roster()
                node = game.add_variation(board.move_stack[0])
                for m in board.move_stack[1:]:
                    node = node.add_variation(m)
                print(game)

        print('Outcome: %s' % board.result())


if __name__ == '__main__':
    main(sys.argv[1:])
