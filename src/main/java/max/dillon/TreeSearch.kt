package max.dillon

import com.google.protobuf.ByteString
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import java.io.FileOutputStream
import java.io.OutputStream
import java.lang.Math.pow
import java.util.*

fun valueFor(player: GameState, node: GameState): Float {
    val sign = if (player.whiteMove == node.whiteMove) 1 else -1
    return sign * node.totalValue
}

fun Float.f3(): String = String.format("%.3f", this)

fun treeSearch(playerState: GameState, temperature: Double): GameState {
    assert(playerState.outcome == GameOutcome.UNDETERMINED)
    val parents = ArrayList<GameState>()
    playerState.expand()
    if (playerState.nextMoves.size == 1) {
        return playerState.nextMoves[0]
    }

    var maxExpansion = if (playerState.model == null) 5000 else 2000
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
    val dist = cumulativeDist(playerState.nextMoves, temperature)
    val r = rand.nextFloat()
    val idx = dist.filter { it < r }.count()
    return playerState.nextMoves[idx]
}

fun cumulativeDist(states: ArrayList<GameState>, temperature: Double): DoubleArray {
    val visits = DoubleArray(states.size) { states[it].visitCount.toDouble() }
    val raised = DoubleArray(states.size) { pow(visits[it], temperature) }
    val rsum = raised.sum()
    var cuml = 0.0
    return DoubleArray(states.size) { cuml += raised[it]; cuml / rsum }
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

fun play(spec: GameGrammar.GameSpec, outputStream: OutputStream, modelFile: String?) {
    var model: MultiLayerNetwork? = null
    if (modelFile != null) {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    }
    var state = GameState(spec, model)
    val stateArray: ArrayList<SlimState> = arrayListOf(slim(state))

    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
//        state = if (state.whiteMove) treeSearch(state) else humanInput(state)
        state = treeSearch(state, 3.0)
        println("${state.moveDepth}: ${state.description}")
        state.printBoard()

        stateArray.add(slim(state))
    }
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
                white = treeSearch(white, 1.0)
                black = sync(black, white)
            } else {
                black = treeSearch(black, 1.0)
                white = sync(white, black)
            }
            println("${white.moveDepth}: ${white.description}")
            white.printBoard()
        }
        when (white.outcome) {
            GameOutcome.WIN_WHITE -> nWhite += 1
            GameOutcome.WIN_BLACK -> nBlack += 1
            GameOutcome.DRAW -> nDraw += 1
        }

        println("White: ${nWhite} Black: ${nBlack} Draw: ${nDraw}")
    }
}
