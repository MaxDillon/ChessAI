package max.dillon

import com.google.protobuf.ByteString
import org.amshove.kluent.`should be in`
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import java.io.FileOutputStream
import java.io.OutputStream
import java.lang.Math.pow
import kotlin.math.sqrt
import java.util.*

fun Float.f3(): String = String.format("%.3f", this)

// returns the value of the state *TO ITSELF*
fun treeSearchSelfValue(state: GameState, counter: ()->Unit): Float {
    if (state.outcome != GameOutcome.UNDETERMINED) {
        counter()
        return state.selfValue()
    }
    if (!state.visited) {
        state.visited = true
        val (selfValue, policy) = state.predict()
        val policySum = state.nextMoves.fold(0f) {
            a, b -> a + policy[b.getMoveIndex()]
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

fun treeSearchMove(state: GameState, temperature: Double): GameState {
    assert(state.outcome == GameOutcome.UNDETERMINED)

    if (state.nextMoves.size == 1) {
        return state.nextMoves[0]
    }

    var i = 5000
    while (i > 0) {
        treeSearchSelfValue(state, { i-- })
    }

    var sum_raised = 0.0
    val raised = DoubleArray(state.nextMoves.size) {
        pow(state.mcts_N[it].toDouble(), 1/temperature).also { sum_raised += it }
    }
    val r = rand.nextFloat()
    var sum_normed = 0.0
    var idx = raised.mapIndexed { j, v ->
        val norm = v / sum_raised
        sum_normed += norm
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

fun humanInput(state: GameState): GameState {
    while (true) {
        print("Enter your move in the same form as description: ")
        val input = readLine()
        state.nextMoves.forEach { if (it.description == input) return it }
        println("that move does not fit the correct form")
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

fun play(spec: GameGrammar.GameSpec, outputStream: OutputStream, modelFile: String?) {
    var model: MultiLayerNetwork? = null
    if (modelFile != null) {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    }
    var state = GameState(spec, model)
    val stateArray: ArrayList<SlimState> = arrayListOf(slim(state))

    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
//        state = if (state.whiteMove) treeSearch(state) else humanInput(state)
        var next = treeSearchMove(state, 1.0)

        for (i in 0 until state.nextMoves.size) {
            println("${state.nextMoves[i].description} ${state.mcts_P[i]} ${state.mcts_N[i]} ${state.mcts_V[i]} ${state.mcts_Pi[i]}")
        }

        if (state.nextMoves.size > 0 && state.nextMoves[0].pi > 0) stateArray.add(slim(state))
        state = next

        println("${state.moveDepth}: ${state.description}")
        state.printBoard()
    }
    println(state.outcome)
    recordGame(state, stateArray, outputStream)
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

fun tournament(spec: GameGrammar.GameSpec, mWhite: String, mBlack: String) {
    val modelWhite = ModelSerializer.restoreMultiLayerNetwork(mWhite)
    val modelBlack = ModelSerializer.restoreMultiLayerNetwork(mBlack)

    var nWhite = 0
    var nBlack = 0
    var nDraw = 0

    while (true) {
        var white = GameState(spec, modelWhite)
        var black = GameState(spec, modelBlack)

        while (white.outcome == GameOutcome.UNDETERMINED) {
            if (white.whiteMove) {
                white = treeSearchMove(white, 1.0)
                black = sync(black, white)
            } else {
                black = treeSearchMove(black, 1.0)
                white = sync(white, black)
            }
            println("${white.moveDepth}: ${white.description}")
            white.printBoard()
        }
        when (white.outcome) {
            GameOutcome.WIN_WHITE -> nWhite += 1
            GameOutcome.WIN_BLACK -> nBlack += 1
            GameOutcome.DRAW -> nDraw += 1
            else -> {}
        }

        println("White: ${nWhite} Black: ${nBlack} Draw: ${nDraw}")
    }
}
