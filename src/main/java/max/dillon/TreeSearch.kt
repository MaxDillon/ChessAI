package max.dillon

import com.google.protobuf.ByteString
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import java.io.OutputStream
import java.lang.Math.pow
import java.util.*
import kotlin.math.sqrt

fun Float.f3(): String = String.format("%.3f", this)

// returns the value of the state *TO ITSELF*
fun treeSearchSelfValue(state: GameState, counter: () -> Unit): Float {
    if (state.outcome != GameOutcome.UNDETERMINED) {
        counter()
        return state.selfValue()
    }
    if (!state.visited) {
        state.visited = true
        val (selfValue, policy) = state.predict()
        state.value = selfValue
        val policySum = state.nextMoves.fold(0f) { a, b ->
            a + policy[b.getMoveIndex()]
        }
        for (i in 0 until state.nextMoves.size) {
            state.mcts_P[i] = policy[state.nextMoves[i].getMoveIndex()] / policySum
        }
        counter()
        if (selfValue != 0f) return selfValue
    }
    val n_sum = state.mcts_N.sum().toDouble() // maybe track as a property
    val next_i = state.nextMoves.indices.maxBy {
        val q = if (state.mcts_N[it] > 0) state.mcts_V[it] / state.mcts_N[it] else 0f
        val u = 2f * state.mcts_P[it] * sqrt(n_sum).toFloat() / (1f + state.mcts_N[it])
        q + u
    } ?: 0
    val next = state.nextMoves[next_i]

    val sign = if (state.whiteMove == next.whiteMove) 1 else -1
    val v_next = treeSearchSelfValue(next, counter) * sign
    state.mcts_V[next_i] += v_next
    state.mcts_N[next_i] += 1
    return v_next
}

var sum_inner = 0.0
var num_inner = 0

fun treeSearchMove(state: GameState, temperature: Double, iter: Int = 1600): GameState {
    assert(state.outcome == GameOutcome.UNDETERMINED)
    if (state.nextMoves.size == 1) {
        return state.nextMoves[0]
    }
    var i = iter
    while (i > 0) {
        treeSearchSelfValue(state, { i-- })
    }
    // if there's a winning move always take it
    for (next in state.nextMoves) {
        var outcome = next.outcome
        if ((state.whiteMove && outcome == GameOutcome.WIN_WHITE) ||
                (!state.whiteMove && outcome == GameOutcome.WIN_BLACK)) {
            return next
        }
    }
    // compute vector of counts raised to power 1/temp
    var sum_raised = 0.0
    val raised = DoubleArray(state.nextMoves.size) {
        pow(state.mcts_N[it].toDouble(), 1 / temperature).also { sum_raised += it }
    }
    // normalize to a probability distribution, and sample a choice from it
    val r = rand.nextFloat()
    var sum_normed = 0.0
    var idx = raised.mapIndexed { j, v ->
        val norm = v / sum_raised
        sum_normed += norm
        sum_inner += state.mcts_P[j] * norm // also keep track of similarity to prior
        num_inner += 1
        state.mcts_Pi[j] = norm.toFloat()
        sum_normed
    }.filter { it < r }.count()
    return state.nextMoves[idx]
}

fun recordGame(finalState: GameState, states: ArrayList<SlimState>, outputStream: OutputStream) {
    states.forEach { state ->
        Instance.TrainingInstance.newBuilder().apply {
            boardState = ByteString.copyFrom(state.state)
            whiteMove = state.whiteMove
            gameLength = finalState.moveDepth
            outcome = when (finalState.outcome) {
                GameOutcome.WIN_WHITE -> if (whiteMove) 1 else -1
                GameOutcome.WIN_BLACK -> if (whiteMove) -1 else 1
                GameOutcome.DRAW -> 0
                else -> throw(RuntimeException("undetermined state at end of game"))
            }
            state.treeSearchResults.forEach {
                addTreeSearchResult(it)
            }
        }.build().writeDelimitedTo(outputStream)
    }
}

data class SlimState(val state: ByteArray,
                     val whiteMove: Boolean,
                     val treeSearchResults: Array<Instance.TreeSearchResult>)

fun slim(state: GameState): SlimState {
    val sz = state.gameSpec.boardSize
    val arr = ByteArray(sz * sz) {
        val x = it / sz
        val y = it % sz
        (state.at(x, y)).toByte()
    }
    val tsr = Array(state.nextMoves.size) {
        val child = state.nextMoves[it]
        Instance.TreeSearchResult.newBuilder().apply {
            index = child.getMoveIndex()
            prob = child.pi
        }.build()
    }
    return SlimState(arr, state.whiteMove, tsr)
}

fun play(spec: GameGrammar.GameSpec, outputStream: OutputStream, modelFile: String?, iter: Int) {
    var model: ComputationGraph? = null
    if (modelFile != null && modelFile != "mcts") {
        model = ModelSerializer.restoreComputationGraph(modelFile)
    }
    var state = GameState(spec, model)
    val stateArray: ArrayList<SlimState> = arrayListOf(slim(state))

    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
        var next = treeSearchMove(state, 1.0, iter)

        for (i in 0 until state.nextMoves.size) {
            println("${state.nextMoves[i].description} ${state.mcts_P[i]} ${state.mcts_N[i]} ${state.mcts_Pi[i]}")
        }

        if (state.nextMoves.size > 0 && state.nextMoves[0].pi > 0) stateArray.add(slim(state))
        state = next

        println("${state.moveDepth}: ${state.description}")
        state.printBoard()
    }
    println(state.outcome)
    recordGame(state, stateArray, outputStream)
    println("Inner Product Stats: ${num_inner} ${sum_inner} ${sum_inner / num_inner}")
}

fun sync(state: GameState, move: GameState): GameState {
    for (next in state.nextMoves) {
        if (next.x1 == move.x1 && next.y1 == move.y1) {
            if (next.x2 == move.x2 && next.y2 == move.y2) {
                return next
            }
        }
    }
    throw RuntimeException("wtf")
}

fun choose(state: GameState): GameState {
    fun parse(s: String) = Pair(s.first() - 'a', s.substring(1).toInt() - 1)
    while (true) {
        println("Enter move (e.g., 'a1' to place or 'a1:b1' to move):")
        val str = readLine()
        if (str == null || str == "") continue
        val tokens = str.split(":")
        if (tokens.size > 2) continue
        val (x1, y1) = if (tokens.size == 1) Pair(-1, -1) else parse(tokens[0])
        val (x2, y2) = parse(tokens[tokens.size - 1])
        for (move in state.nextMoves) {
            if (move.x2 == x2 && move.y2 == y2 &&
                    (move.x1 == x1 || x1 < 0) && (move.y1 == y1 || y1 < 0)) {
                return move
            }
        }
        println("Try again")
    }
}

fun tournament(spec: GameGrammar.GameSpec, white: String, black: String, iter: Int,
               outputStream: OutputStream) {
    var modelWhite: ComputationGraph? = null
    var modelBlack: ComputationGraph? = null

    if (white != "human" && white != "mcts") {
        modelWhite = ModelSerializer.restoreComputationGraph(white)
    }
    if (black != "human" && black != "mcts") {
        modelBlack = ModelSerializer.restoreComputationGraph(black)
    }

    var nWhite = 0
    var nBlack = 0
    var nDraw = 0

    while (true) {
        var whiteState = GameState(spec, modelWhite)
        var blackState = GameState(spec, modelBlack)
        val stateArray: ArrayList<SlimState> = arrayListOf(slim(whiteState))

        if (white == "human" || black == "human") {
            whiteState.printBoard()
        }
        while (whiteState.outcome == GameOutcome.UNDETERMINED) {
            if (whiteState.whiteMove) {
                whiteState = if (white == "human") {
                    choose(whiteState)
                } else {
                    var next = treeSearchMove(whiteState, 0.8, iter)
                    if (whiteState.nextMoves.size > 0 && whiteState.nextMoves[0].pi > 0) {
                        stateArray.add(slim(whiteState))
                    }
                    next
                }
                blackState = sync(blackState, whiteState)
            } else {
                blackState = if (black == "human") {
                    choose(blackState)
                } else {
                    var next = treeSearchMove(blackState, 0.8, iter)
                    if (blackState.nextMoves.size > 0 && blackState.nextMoves[0].pi > 0) {
                        stateArray.add(slim(blackState))
                    }
                    next
                }
                whiteState = sync(whiteState, blackState)
            }
            println("${whiteState.moveDepth}: ${whiteState.description}")
            whiteState.printBoard()
        }
        when (whiteState.outcome) {
            GameOutcome.WIN_WHITE -> nWhite += 1
            GameOutcome.WIN_BLACK -> nBlack += 1
            GameOutcome.DRAW -> nDraw += 1
            else -> {
            }
        }
        println("White: ${nWhite} Black: ${nBlack} Draw: ${nDraw}")
        recordGame(whiteState, stateArray, outputStream)
    }
}
