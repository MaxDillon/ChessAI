package max.dillon

import com.google.protobuf.ByteString
import java.io.FileOutputStream
import java.io.OutputStream
import java.lang.Math.pow
import java.util.*

fun valueFor(player: GameState, node: GameState): Float {
    val sign = if (player.whiteMove == node.whiteMove) 1 else -1
    return sign * node.totalValue
}

fun Float.f3(): String = String.format("%.3f", this)

fun treeSearch(playerState: GameState): GameState {
    assert(playerState.outcome == GameOutcome.UNDETERMINED)
    val parents = ArrayList<GameState>()
    playerState.expand()
    if (playerState.nextMoves.size == 1) {
        return playerState.nextMoves[0]
    }

    var maxExpansion = 5000
    while (maxExpansion > 0) {
        parents.clear()
        var currentNode = playerState

        while (true) {
            parents.add(currentNode)
            currentNode = currentNode.nextMoves.maxBy {
                it.scoreFor(currentNode)
            } ?: throw RuntimeException("wtf")
            if (!currentNode.leaf && currentNode.outcome == GameOutcome.UNDETERMINED) {
                continue
            }
            maxExpansion--
            currentNode.expand() // make sure we have node value evaluated
            if (currentNode.outcome != GameOutcome.UNDETERMINED) {
                parents.forEach { it.updateValue(valueFor(it, currentNode)) }
                break
            }
            if (currentNode.totalValue == 0f) {
                continue
            } else {
                parents.forEach { it.updateValue(valueFor(it, currentNode)) }
                break
            }
        }

    }
    val dist = cumulativeDist(playerState.nextMoves)
    val r = rand.nextFloat()
    val idx = dist.filter { it < r }.count()
    return playerState.nextMoves[idx]
}

val temperature = 3.0

fun cumulativeDist(states: ArrayList<GameState>): DoubleArray {
    val visits = DoubleArray(states.size, { states[it].visitCount.toDouble() })
    val raised = DoubleArray(states.size, { pow(visits[it], temperature) })
    val rsum = raised.sum()
    var cuml = 0.0
    return DoubleArray(states.size, { cuml += raised[it]; cuml / rsum })
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
            prior = child.prior
            meanValue = child.totalValue / child.visitCount
            numVisits = child.visitCount.toFloat()
        }.build()
    }
    return SlimState(arr, state.whiteMove, tsr)
}

fun play(spec: GameGrammar.GameSpec, outputStream: OutputStream) {
    var state = GameState(spec)
    val stateArray: ArrayList<SlimState> = arrayListOf(slim(state))

    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
//        state = if (state.whiteMove) treeSearch(state) else humanInput(state)
        state = treeSearch(state)
        println("${state.moveDepth}: ${state.description}")
        state.printBoard()

        stateArray.add(slim(state))
    }
    recordGame(state, stateArray, outputStream)
}

