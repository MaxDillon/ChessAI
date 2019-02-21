package maximum.industries

import maximum.industries.games.ChessState
import org.nd4j.linalg.factory.Nd4j
import java.lang.RuntimeException
import java.util.*

fun uciLoop(gameSpec: GameGrammar.GameSpec, algo: GameSearchAlgo) {

    val stack = ArrayList<ChessState>()

    fun uci() {
        println("id name yacej")
        println("uciok")
    }

    fun isready() {
        println("readyok")
    }

    fun ucinewgame() {
        algo.gameOver() // clear mcts caches
        stack.clear()
        stack.add(newGame(gameSpec) as ChessState)
    }

    val startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - - 1"

    fun position(toks: List<String>) {
        // We'll be given a list of tokens in one of these forms:
        //   startpos
        //   startpos moves e2e4 e7e5 ...
        //   fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        //   fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4 ...
        // Possible that "moves" may be omitted, so we'll allow it to be present or absent.
        var fen = startingFen
        var pos = 1
        if (toks[0] == "fen") {
            // strip out en passant and half-move count, which we don't support
            fen = "${toks[1]} ${toks[2]} ${toks[3]} - - ${toks[6]}"
            pos = 7
        }
        if (toks.size > pos && toks[pos] == "moves") {
            pos++
        }
        for (i in 0 until stack.size) {
            if (stack[i].fen() == fen) {
                while (stack.size > i + 1) stack.removeAt(stack.size - 1)
                // state with given fen is now at the top of the stack
                // now attempt to apply move list
                for (j in pos until toks.size) {
                    val moveUci = toks[j].substring(0, 4) // strip promotion info
                    var found = false
                    for (c in stack.last().nextMoves) {
                        val candidate = c as ChessState
                        val candidateUci = candidate.uci().substring(0, 4) // strip promotion
                        // in the event of underpromotion we may desynchronize the board
                        // possible that we'll resync later after an irreversible move that
                        // restarts the position from a new fen. but there is a risk in the
                        // meantime that the opponent will make an illegal move.
                        if (moveUci == candidateUci) {
                            stack.add(candidate)
                            found = true
                            break
                        }
                    }
                    if (!found) throw RuntimeException("Cannot apply position")
                }
                return // position successfully applied
            }
        }
        // did not find state with given fen, but it may be a child of the last node.
        for (c in stack.last().nextMoves) {
            val candidate = c as ChessState
            if (fen == candidate.fen()) {
                stack.add(candidate)
                return // position successfully applied
            }
        }
        // could not find position in existing node tree, so throw away state and restart.
        // in principle this arises with new games, or when an illegal move (from the
        // perspective of chess2) is played. e.g., an en-passant capture or a knight move
        // after an underpromotion. In the latter cases we'll only succeed in resyncing if
        // the client restarts with a fresh board from the given fen instead of sending a
        // move sequence.
        stack.clear()
        stack.add(ChessState.fromFen(gameSpec, fen))
        position(toks)  // should be safe. we are guaranteed to find the fen
    }

    fun go() {
        val (nextState, _) = algo.next(stack.last())
        stack.add(nextState as ChessState)
        println("bestmove ${nextState.uci()}")
    }

    while (true) {
        var toks = readLine()!!.split(' ')
        val command = toks[0]
        toks = toks.slice(1 until toks.size)
        when (command) {
            "uci" -> uci()
            "isready" -> isready()
            "ucinewgame" -> ucinewgame()
            "position" -> position(toks)
            "go" -> go()
            "quit" -> System.exit(0)
        }
    }
}

fun main(args: Array<String>) {
    val model = getArg(args, "model") ?: "model2:prod_model.chess2"
    val device = getArg(args, "device")?.toInt() ?: 0
    val seed = getArg(args, "seed")?.toLong() ?: 0L
    if (seed != 0L) rand = Random(seed)
    Nd4j.getAffinityManager().attachThreadToDevice(Thread.currentThread(), device)
    Nd4j.getMemoryManager().autoGcWindow = 10000

    val gameSpec = loadSpec("chess2")
    val uParams = getSearchParameters(args, "")
    val uciAlgo = getAlgo(model, uParams)
    uciLoop(gameSpec, uciAlgo)
}
