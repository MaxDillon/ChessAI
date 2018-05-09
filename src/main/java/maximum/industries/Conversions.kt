package maximum.industries

import com.google.protobuf.ByteString
import maximum.industries.GameGrammar.GameSpec
import maximum.industries.GameGrammar.MoveSource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.abs

fun GameSpec.numRealPieces() = pieceCount - 1

// should only be called for nonzero pieces. zero means no piece present.
fun GameSpec.pieceToChannel(piece: Int) =
        if (piece > 0) piece else numRealPieces() + (-piece)

// should only be called for nonzero channels. zero means channel indicating turn.
fun GameSpec.channelToPiece(channel: Int) =
        if (channel <= numRealPieces()) channel else -(channel - numRealPieces())

fun GameSpec.indexToXy(index: Int) =
        Pair(index % boardSize, index / boardSize) // row * size + col

fun GameSpec.xyToIndex(xy: Pair<Int, Int>) =
        xy.second * boardSize + xy.first  // row * size + col

data class MoveInfo(var x1: Int, var y1: Int, var x2: Int, var y2: Int, var p1: Int) {
    override fun toString() = "${'a' + x1}${y1 + 1} -> ${'a' + x2}${y2 + 1}"
}

fun GameSpec.toPolicyIndex(info: MoveInfo): Int {
    val dst = xyToIndex(Pair(info.x2, info.y2))
    val src = if (moveSource == MoveSource.ENDS) {
        abs(info.p1) - 1 // subtract out the virtual piece added to all gamespecs
    } else {
        xyToIndex(Pair(info.x1, info.y1))
    }
    return src * boardSize * boardSize + dst
}

fun GameSpec.expandPolicyIndex(index: Int): MoveInfo {
    val dst = index % (boardSize * boardSize)
    val (x2, y2) = indexToXy(dst)
    val src = index / (boardSize * boardSize)
    if (moveSource == MoveSource.ENDS) {
        return MoveInfo(0, 0, x2, y2, src + 1)
    } else {
        val (x1, y1) = indexToXy(src)
        return MoveInfo(x1, y1, x2, y2, 0)
    }
}

fun GameSpec.flip(xy: Pair<Int, Int>, leftRight: Boolean, sides: Boolean): Pair<Int, Int> {
    val x = if (leftRight) boardSize - 1 - xy.first else xy.first
    val y = if (sides) boardSize - 1 - xy.second else xy.second
    return Pair(x, y)
}

fun GameSpec.flipPolicyIndex(index: Int, leftRight: Boolean, sides: Boolean): Int {
    val (x1i, y1i, x2i, y2i, p1) = expandPolicyIndex(index)
    val (x2, y2) = flip(Pair(x2i, y2i), leftRight, sides)
    if (moveSource == MoveSource.ENDS) {
        return toPolicyIndex(MoveInfo(0, 0, x2, y2, p1))
    } else {
        val (x1, y1) = flip(Pair(x1i, y1i), leftRight, sides)
        return toPolicyIndex(MoveInfo(x1, y1, x2, y2, p1))
    }
}

fun GameState.toPolicyIndex(): Int {
    return gameSpec.toPolicyIndex(MoveInfo(x1, y1, x2, y2, p1))
}

fun GameState.toModelInput(): INDArray {
    val size = gameSpec.boardSize
    val input = Nd4j.zeros(1, 2 * gameSpec.numRealPieces() + 1, size, size)
    for (x in 0 until size) {
        for (y in 0 until size) {
            val p = at(x, y)
            if (p != 0) {
                input.putScalar(intArrayOf(0, gameSpec.pieceToChannel(p), y, x), 1f)
            }
        }
    }
    val turn = Nd4j.ones(size, size)
    if (player.eq(Player.BLACK)) turn.muli(-1)
    input.put(arrayOf(NDArrayIndex.point(0), NDArrayIndex.point(0)), turn)
    return input
}

fun GameSpec.fromModelInput(input: INDArray, batchIndex: Int): GameState {
    val size = boardSize
    val newBoard = Array(size) { IntArray(size) { 0 } }
    for (channel in 1..(2 * numRealPieces())) {
        for (x in 0 until size) {
            for (y in 0 until size) {
                if (input.getFloat(intArrayOf(batchIndex, channel, y, x)) > 0) {
                    newBoard[y][x] = channelToPiece(channel)
                }
            }
        }
    }
    val player = if (input.getFloat(intArrayOf(batchIndex, 0, 0, 0)) > 0) Player.WHITE else Player.BLACK
    return GameState(this, newBoard, player, 0, 0, 0, 0, 0, 0)
}

fun GameState.toSlimState(initTsr: (i: Int, builder: Instance.TreeSearchResult.Builder) -> Unit): SlimState {
    val size = gameSpec.boardSize
    val array = ByteArray(size * size) {
        val (x, y) = gameSpec.indexToXy(it)
        at(x, y).toByte()
    }
    val tsrs = Array(nextMoves.size) {
        Instance.TreeSearchResult.newBuilder().apply {
            index = nextMoves[it].toPolicyIndex()
            initTsr(it, this)
        }.build()
    }
    val player = if (player.eq(Player.WHITE)) {
        Instance.Player.WHITE
    } else {
        Instance.Player.BLACK
    }
    return SlimState(array, player, tsrs)
}

fun SlimState.toTrainingInstance(finalOutcome: Int, finalDepth: Int): Instance.TrainingInstance {
    val slimPlayer = player // to disambiguate below
    return Instance.TrainingInstance.newBuilder().apply {
        boardState = ByteString.copyFrom(state)
        player = slimPlayer
        gameLength = finalDepth
        outcome = finalOutcome
        treeSearchResults.forEach { addTreeSearchResult(it) }
    }.build()
}

fun SlimState.toTrainingInstance(finalState: GameState): Instance.TrainingInstance {
    val slimWhite = player.eq(Instance.Player.WHITE)
    val finalWhite = finalState.player.eq(Player.WHITE)
    val outcome = when (finalState.outcome) {
        Outcome.WIN -> if (slimWhite == finalWhite) 1 else -1
        Outcome.LOSE -> if (slimWhite == finalWhite) -1 else 1
        Outcome.DRAW -> 0
        else -> throw(RuntimeException("undetermined state at end of game"))
    }
    return toTrainingInstance(outcome, finalState.moveDepth)
}

fun Instance.TrainingInstance.toBatchTrainingInput(
        gameSpec: GameSpec, batchIndex: Int, reflection: Int,
        input: INDArray, value: INDArray, policy: INDArray, legal: INDArray) {
    val flipLeftRight = reflection % 2 > 0
    val reverseSides = reflection / 2 > 0

    for (i in 0 until boardState.size()) {
        val xy = gameSpec.indexToXy(i)
        val (x, y) = gameSpec.flip(xy, flipLeftRight, reverseSides)

        val p_raw = boardState.byteAt(i).toInt()
        val p = if (reverseSides) -p_raw else p_raw

        if (p != 0) {
            input.putScalar(intArrayOf(batchIndex, gameSpec.pieceToChannel(p), y, x), 1)
        }
    }

    val turn_raw = if (player.eq(Instance.Player.WHITE)) 1 else -1
    val turn = if (reverseSides) -turn_raw else turn_raw
    input.put(arrayOf(NDArrayIndex.point(batchIndex), NDArrayIndex.point(0)), turn)

    for (tsr in treeSearchResultList) {
        val policyIndex = gameSpec.flipPolicyIndex(tsr.index, flipLeftRight, reverseSides)
        policy.putScalar(intArrayOf(batchIndex, policyIndex), tsr.prob)
        legal.putScalar(intArrayOf(batchIndex, policyIndex), 1f)
    }
    value.putScalar(batchIndex, outcome)
}

