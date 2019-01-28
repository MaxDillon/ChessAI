package maximum.industries

import com.google.protobuf.ByteString
import maximum.industries.GameGrammar.GameSpec
import maximum.industries.GameGrammar.MoveSource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.tensorflow.Tensor
import java.lang.Math.pow
import java.nio.FloatBuffer
import kotlin.math.E
import kotlin.math.abs
import kotlin.math.log

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
    val src = if (moveSource == MoveSource.MOVESOURCE_ENDS) {
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
    if (moveSource == MoveSource.MOVESOURCE_ENDS) {
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
    if (moveSource == MoveSource.MOVESOURCE_ENDS) {
        return toPolicyIndex(MoveInfo(0, 0, x2, y2, p1))
    } else {
        val (x1, y1) = flip(Pair(x1i, y1i), leftRight, sides)
        return toPolicyIndex(MoveInfo(x1, y1, x2, y2, p1))
    }
}

fun GameState.toPolicyIndex(): Int {
    return gameSpec.toPolicyIndex(MoveInfo(x1, y1, x2, y2, p1))
}

fun GameState.toModelInput(reflections: IntArray = intArrayOf(0)): INDArray {
    val size = gameSpec.boardSize
    val input = Nd4j.zeros(reflections.size, 2 * gameSpec.numRealPieces() + 1, size, size)
    for (i in 0 until reflections.size) {
        toModelInput(input, i.toLong(), reflections[i])
    }
    return input
}

// NOTE: the input INDArray is assumed to be zero-initialized.
fun GameState.toModelInput(input: INDArray, batchIndex: Long = 0, reflection: Int = 0) {
    val flipLeftRight = reflection % 2 > 0
    val reverseSides = reflection / 2 > 0
    val size = gameSpec.boardSize
    for (x in 0 until size) {
        for (y in 0 until size) {
            val p_raw = at(x, y)
            if (p_raw != 0) {
                val (fx, fy) = gameSpec.flip(Pair(x,y), flipLeftRight, reverseSides)
                val p = if (reverseSides) -p_raw else p_raw
                input.putScalar(longArrayOf(batchIndex, gameSpec.pieceToChannel(p).toLong(), fy.toLong(), fx.toLong()), 1f)
            }
        }
    }
    val turn_raw = if (player.eq(Player.WHITE)) 1 else -1
    val turn = if (reverseSides) -turn_raw else turn_raw
    input.put(arrayOf(NDArrayIndex.point(batchIndex), NDArrayIndex.point(0)), turn)
}

fun GameState.toTensorInput(): Tensor<Float> {
    val turn = if (player.eq(Player.WHITE)) 1 else -1
    val size = gameSpec.boardSize
    val channels = gameSpec.numRealPieces() * 2 + 1
    val buf = FloatBuffer.allocate(1 * channels * size * size)
    for (x in 0 until size) {
        for (y in 0 until size) {
            val p_raw = at(x, y)
            if (p_raw != 0) {
                val p = gameSpec.pieceToChannel(p_raw)
                buf.put(p * size * size + y * size + x, 1f)
            }
            buf.put(y * size + x, turn.toFloat())
        }
    }
    return Tensor.create(longArrayOf(1, channels.toLong(), size.toLong(), size.toLong()), buf)
}

fun GameSpec.fromTensorInput(tensor: Tensor<Float>): GameState {
    val size = boardSize
    val newBoard = ByteArray(size * size) { 0 }
    val buf = FloatBuffer.allocate((1 + 2 * numRealPieces()) * size * size)
    tensor.writeTo(buf)
    for (channel in 1..(2 * numRealPieces())) {
        for (x in 0 until size) {
            for (y in 0 until size) {
                val p = buf.get(channel * size * size + y * size + x)
                if (p > 0) {
                    newBoard[y * size + x] = channelToPiece(channel).toByte()
                }
            }
        }
    }
    val player = if (buf.get(0) > 0) Player.WHITE else Player.BLACK
    return StateFactory.newState(this, newBoard, player,
                                 0, 0, 0, 0, 0, 0)
}

fun GameSpec.fromModelInput(input: INDArray, batchIndex: Int = 0): GameState {
    val size = boardSize
    val newBoard = ByteArray(size * size) { 0 }
    for (channel in 1..(2 * numRealPieces())) {
        for (x in 0 until size) {
            for (y in 0 until size) {
                if (input.getFloat(intArrayOf(batchIndex, channel, y, x)) > 0) {
                    newBoard[y * size + x] = channelToPiece(channel).toByte()
                }
            }
        }
    }
    val player = if (input.getFloat(intArrayOf(batchIndex, 0, 0, 0)) > 0) Player.WHITE else Player.BLACK
    return StateFactory.newState(this, newBoard, player,
                                 0, 0, 0, 0, 0, 0)
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

fun SlimState.toTrainingInstance(finalOutcome: Int, finalDepth: Short): Instance.TrainingInstance {
    val slimPlayer = player // to disambiguate below
    return Instance.TrainingInstance.newBuilder().apply {
        boardState = ByteString.copyFrom(state)
        player = slimPlayer
        gameLength = finalDepth.toInt()
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

fun GameState.legalMovesToList(): IntArray {
    val moves = arrayListOf<Int>()
    for (move in nextMoves) {
        moves.add(gameSpec.xyToIndex(Pair(move.x1, move.y1)))
        moves.add(gameSpec.xyToIndex(Pair(move.x2, move.y2)))
    }

    return moves.toIntArray()
}

fun WireState.toGameState(spec: GameSpec): GameState {
    val player = if (state.whiteMove) Player.WHITE else Player.BLACK
    return StateFactory.newState(spec, state.board, player,
                                 0, -1, -1, -1, -1,
                                 state.moveDepth.toShort())
}

fun GameState.toWireState(): WireState {
    val obj = ObjectState(gameBoard, legalMovesToList(), player.eq(Player.WHITE), moveDepth.toInt())
    return WireState(obj, -1, obj.hashCode())
}

fun normalize(probs: FloatArray) {
    var sum = probs.sum()
    for (i in 0 until probs.size) probs[i] /= sum
}

fun entropy(probs: FloatArray): Float {
    var ent = 0f
    for (p in probs) ent -= p * log(p, E.toFloat())
    return ent
}

fun raise(probs: FloatArray) {
    for (i in 0 until probs.size) probs[i] = pow(probs[i].toDouble(), 1.5).toFloat()
}

fun addeps(probs: FloatArray) {
    for (i in 0 until probs.size) probs[i] += 0.001f
}

fun adjustEntropy(probs: FloatArray, metf: Double) {
    if (metf <= 0) return
    val maxEntropy = -((1 - metf) * log((1 - metf) / probs.size, E) - metf * log(metf, E))
    addeps(probs)
    normalize(probs)
    var maxIter = 15;
    while (entropy(probs) > maxEntropy && maxIter-- > 0) {
        raise(probs)
        addeps(probs)
        normalize(probs)
    }
}

fun Instance.TrainingInstance.toBatchTrainingInput(
        gameSpec: GameSpec, batchIndex: Long, reflection: Int,
        input: INDArray, value: INDArray, policy: INDArray, legal: INDArray,
        valueMult: Float = 1.0f, maxEntropyTopFrac: Double = 0.0) {

    val flipLeftRight = reflection % 2 > 0
    val reverseSides = reflection / 2 > 0
    for (i in 0 until boardState.size()) {
        val p_raw = boardState.byteAt(i).toInt()
        if (p_raw != 0) {
            val xy = gameSpec.indexToXy(i)
            val (x, y) = gameSpec.flip(xy, flipLeftRight, reverseSides)
            val p = if (reverseSides) -p_raw else p_raw
            input.putScalar(longArrayOf(batchIndex, gameSpec.pieceToChannel(p).toLong(), y.toLong(), x.toLong()), 1)
        }
    }
    val turn_raw = if (player.eq(Instance.Player.WHITE)) 1 else -1
    val turn = if (reverseSides) -turn_raw else turn_raw
    input.put(arrayOf(NDArrayIndex.point(batchIndex), NDArrayIndex.point(0)), turn)

    val probs = FloatArray(treeSearchResultCount) { treeSearchResultList[it].prob }
    adjustEntropy(probs, maxEntropyTopFrac)

    for (i in 0 until treeSearchResultCount) {
        val policyIndex = gameSpec.flipPolicyIndex(
                treeSearchResultList[i].index, flipLeftRight, reverseSides).toLong()
        policy.putScalar(longArrayOf(batchIndex, policyIndex), probs[i])
        legal.putScalar(longArrayOf(batchIndex, policyIndex), 1f)
    }
    value.putScalar(batchIndex, outcome * valueMult)
}

class StateFactory {
    companion object {
        fun newGame(gameSpec: GameGrammar.GameSpec): GameState {
            return if (gameSpec.implementingClass == "") {
                GameState(gameSpec)
            } else {
                Class.forName(gameSpec.implementingClass)
                        .getConstructor(gameSpec.javaClass)
                        .newInstance(gameSpec) as GameState
            }
        }

        fun newState(gameSpec: GameSpec, gameBoard: ByteArray, player: Player,
                    p1: Int, x1: Int, y1: Int, x2: Int, y2: Int, moveDepth: Short,
                    history: IntArray = IntArray(16) { 0 }): GameState {
            return if (gameSpec.implementingClass == "") {
                GameState(gameSpec, gameBoard, player, p1, x1, y1, x2, y2, moveDepth, history)
            } else {
                Class.forName(gameSpec.implementingClass)
                        .getConstructor(gameSpec.javaClass,
                                        gameBoard.javaClass,
                                        player.javaClass,
                                        p1.javaClass,
                                        x1.javaClass,
                                        y1.javaClass,
                                        x2.javaClass,
                                        y2.javaClass,
                                        moveDepth.javaClass,
                                        history.javaClass)
                        .newInstance(gameSpec, gameBoard, player, p1, x1, y1, x2, y2,
                                     moveDepth, history) as GameState
            }
        }
    }
}
