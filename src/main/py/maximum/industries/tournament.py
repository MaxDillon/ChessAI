import sys
import chess
import chess.engine


def get_yace_java_engine(model='tf:tfmodels/r14/1549380964', args='-iter 200 -temp 0.1'):
    yace = chess.engine.SimpleEngine.popen_uci(
        ['./mi_uci.sh',
         '-model', model,
         ] + args.split(' '))
    return yace, chess.engine.Limit(), {}


def get_yace_python_engine(model='tfmodels/r14/1549380964', args='iter=200,temp=0.1'):
    yace = chess.engine.SimpleEngine.popen_uci(
        ['/usr/local/bin/python', '-u', 'src/main/py/maximum/industries/play.py',
         '--uci',
         '--model=%s' % model,
         '--uargs=%s' % args
         ])
    return yace, chess.engine.Limit(), {}


def get_stockfish_engine(level):
    stockfish = chess.engine.SimpleEngine.popen_uci("./stockfish")
    # Per https://github.com/niklasf/fishnet/blob/master/fishnet.py
    # ~elo = [1350, 1420, 1500, 1600, 1700, 1900, 2200, 2600]
    skills = [   0,    3,    6,   10,   14,   16,   18,   20]
    depths = [   1,    1,    2,    3,    5,    8,   13,   22]
    mtimes = [  50,  100,  150,  200,  300,  400,  500, 1000]
    limits = chess.engine.Limit(time=mtimes[level-1], depth=depths[level-1])
    options = { 'Skill Level': skills[level-1] }
    return stockfish, limits, options


def get_engine(engine, model, args):
    if engine == 'java':
        return get_yace_java_engine(model=model, args=args)
    elif engine == 'python':
        return get_yace_python_engine(model=model, args=args)
    elif engine == 'stockfish':
        return get_stockfish_engine(int(args))
    else:
        raise Exception('invalid engine')


def play(white, black):
    board = chess.Board()
    while not board.is_game_over():
        player, limits, options = white if board.turn else black
        result = player.play(board, limits, options=options)
        # will restart from fen and clear move stack after en passant or underpromotion
        # in order to make sure the java engine can stay synchronized.
        need_restart = board.is_en_passant(result.move) or result.move.promotion in [2, 3, 4]
        board.push(result.move)
        if need_restart:
            board = chess.Board(board.fen())
        if board.turn:
            print('.', end='', flush=True)
    print('\033[2K\r', end='')
    return board.result()


def tournament(white, black, n):
    white_score = 0
    for _ in range(n):
        result = play(white, black)
        print(result)
        if result == '1-0':
            white_score += 1
        elif result == '0-1':
            white_score += 0
        else:
            white_score += 0.5
    return white_score / n


def main(argv):
    n = int(argv[0])
    white = get_engine(argv[1], argv[2], argv[3])
    black = get_engine(argv[4], argv[5], argv[6])
    white_score = tournament(white, black, n)
    print('White score: %3.2f' % white_score)
    white[0].quit()
    black[0].quit()
    exit(0)


if __name__ == '__main__':
    main(sys.argv[1:])
