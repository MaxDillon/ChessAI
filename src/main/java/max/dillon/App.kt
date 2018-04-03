package max.dillon

import com.google.protobuf.TextFormat
import max.dillon.GameGrammar.*
import max.dillon.GameGrammar.Outcome.*
import max.dillon.GameGrammar.Symmetry.NONE
import max.dillon.GameGrammar.Symmetry.ROTATE
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.NoSuchFileException
import java.nio.file.Paths
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.*

enum class GameOutcome {
    UNDETERMINED, WIN_WHITE, WIN_BLACK, DRAW
}

class Memoize1<in T, out R>(val f: (T) -> R) : (T) -> R {
    private val values = mutableMapOf<T, R>()
    override fun invoke(x: T): R {
        return values.getOrPut(x, { f(x) })
    }
}

fun <T, R> ((T) -> R).memoize(): (T) -> R = Memoize1(this)

private fun square(size: Int): List<Pair<Int, Int>> {
    val moves = arrayListOf<Pair<Int, Int>>()
    for (i in 0..size) {
        for (j in 1..size) {
            moves.add(Pair(i, j))
            moves.add(Pair(-j, i))
            moves.add(Pair(-i, -j))
            moves.add(Pair(j, -i))
        }
    }
    return moves
}

val square_m = ::square.memoize()

private fun plus(size: Int): List<Pair<Int, Int>> {
    return square(size).filter { it.first == 0 || it.second == 0 }
}

val plus_m = ::plus.memoize()

private fun cross(size: Int): List<Pair<Int, Int>> {
    return square(size).filter { Math.abs(it.first) == Math.abs(it.second) }
}

val cross_m = ::cross.memoize()

val rand = Random()

class GameState {
    var gameBoard: Array<IntArray>
    var rowOwned: BooleanArray
    val gameSpec: GameSpec
    var whiteMove = true
    var x1 = -1
    var y1 = -1
    var x2 = -1
    var y2 = -1
    var description = ""
    var pieceMoved = 0
    var moveDepth = 0
    val nextMoves: ArrayList<GameState> by lazy {
        getLegalNextStates()
    }
    val outcome: GameOutcome by lazy {
        gameOutcome()
    }
    var leaf = true
    var prior = 0f
    var visitCount = 0
    var totalValue = 0.0f
    val model: MultiLayerNetwork?

    constructor(gameSpec: GameSpec, model: MultiLayerNetwork? = null) {
        this.gameSpec = gameSpec
        this.model = model
        gameBoard = Array(gameSpec.boardSize) { IntArray(gameSpec.boardSize) { 0 } }
        rowOwned = BooleanArray(gameSpec.boardSize) { true }
        gameSpec.pieceList.forEachIndexed { pieceType, piece ->
            piece.placementList.forEach { placement ->
                val (x, y) = placement.substring(1).split("y").map { it.toInt() - 1 }
                if ((gameBoard[y][x] == 0)) {
                    gameBoard[y][x] = pieceType
                    val oppositeX =
                            if (gameSpec.boardSymmetry == ROTATE)
                                gameSpec.boardSize - 1 - x else x
                    val oppositeY = gameSpec.boardSize - 1 - y
                    gameBoard[oppositeY][oppositeX] = -pieceType
                } else throw RuntimeException("you cant place a piece at $x,$y")
            }
        }
    }

    constructor(prev: GameState, x1: Int, y1: Int, x2: Int, y2: Int, piece: Int) {
        this.gameSpec = prev.gameSpec
        this.model = prev.model
        this.gameBoard = Array(prev.gameBoard.size) { prev.gameBoard[it] }
        this.rowOwned = BooleanArray(gameSpec.boardSize) { false }
        this.whiteMove = !prev.whiteMove
        this.moveDepth = prev.moveDepth + 1
        this.x1 = x1
        this.y1 = y1
        this.x2 = x2
        this.y2 = y2
        this.pieceMoved = piece
        this.description = "${'a' + x1}${y1 + 1} -> ${'a' + x2}${y2 + 1}"
    }

    override fun toString(): String = "${moveDepth}: after ${description} whitemove: ${whiteMove}"

    fun scoreFor(parent: GameState): Float {
        val tempScore: Float = when (outcome) {
            GameOutcome.UNDETERMINED -> {
                // See https://en.wikipedia.org/wiki/Monte_Carlo_tree_search for
                // formula balancing exploitation/exploration
                // exploration value is independent of prior or mcts estimate
                // and just depends on visit counts
                val explorationValue = sqrt(
                        2f * log(parent.visitCount.toFloat() + 1f, E.toFloat()) /
                                (visitCount.toFloat() + 1f))
                val mctsEstValue = totalValue / (visitCount + 1)
                val priorEstValue = prior
                // TODO: Not sure exactly how to combine these:
                val estValue = (mctsEstValue + priorEstValue) / 2
                val sign = if (parent.whiteMove == whiteMove) 1 else -1
                return (sign * estValue) + explorationValue
            }
            GameOutcome.WIN_WHITE -> {
                if (parent.whiteMove) 10f else -10f
            }
            GameOutcome.WIN_BLACK -> {
                if (parent.whiteMove) -10f else 10f
            }
            GameOutcome.DRAW -> {
                0f
            }
        }
        return tempScore
    }

    fun updateValue(value: Float) {
        totalValue += value
        visitCount += 1
    }

    fun at(x: Int, y: Int): Int {
        return gameBoard[y][x]
    }

    private fun getPiece(i: Int): GameGrammar.Piece = gameSpec.getPiece(abs(i))

    private fun setState(x: Int, y: Int, state: Int) {
        if (!rowOwned[y]) {
            gameBoard[y] = gameBoard[y].clone()
            rowOwned[y] = true
        }
        // y is row, x is column
        gameBoard[y][x] = state
    }

    private fun numEmptyCells(): Int =
            gameBoard.map { it.filter { it == 0 }.count() }.sum()

    /**
     * A method to map a GameState for a given legal move into an output space
     * of size srcSize * dstSize where dstSize = (N x N) where N is the board
     * size and srcSize is either (N x N) or P where P is the number piece
     * classes (to deal with cases where we place a new piece on the board and
     * need to identify its class).
     */
    fun getMoveIndex(): Int {
        val dim = gameSpec.boardSize
        val dst = y2 + dim * x2
        val src = if (gameSpec.moveSource == MoveSource.ENDS) {
            abs(pieceMoved) - 1 // subtract out the virtual piece added to all gamespecs
        } else {
            y1 + x1 * dim
        }
        return src * dim * dim + dst
    }

    fun toModelInput(): INDArray {
        var sz = gameSpec.boardSize
        var np = gameSpec.pieceCount - 1
        var input = Nd4j.zeros(1, 2 * np + 1, sz, sz)
        for (x in 0 until sz) {
            for (y in 0 until sz) {
                val p = at(x,y)
                if (p != 0) {
                    val channel = if (p > 0) p else np - p
                    input.putScalar(intArrayOf(0, channel, x, y), 1)
                }
            }
        }
        var turn = if (whiteMove) 0 else 1
        for (x in 0 until sz) for (y in 0 until sz) input.putScalar(intArrayOf(0, 0, x, y), turn)
        return input
    }

    fun modelPredict(network: MultiLayerNetwork): Pair<Float, FloatArray> {
        val policy = network.output(toModelInput())
        return Pair(0f, FloatArray(policy.shape()[1]) {
            policy.getFloat(intArrayOf(0, it))
        })
    }

    fun predict(): Pair<Float, FloatArray> {
        if (model == null) {
            val dim = gameSpec.boardSize
            val srcsz = if (gameSpec.moveSource == MoveSource.ENDS) gameSpec.pieceCount - 1 else dim * dim
            return Pair(0.0f, FloatArray(srcsz * dim * dim, {
                0.5f + (rand.nextFloat() - 0.5f) / 10
            }))
        } else {
            return modelPredict(model)
        }
    }

    fun checkLandingConstraints(dest: Int, piece: Int, move: Move): Boolean {
        if (dest == 0 && move.land.none == DISALLOWED) {
            return false
        } else if (dest.sign == piece.sign && move.land.own == DISALLOWED) {
            return false
        } else if (dest.sign == -piece.sign && move.land.opponent == DISALLOWED) {
            return false
        }
        return true
    }

    fun offBoard(x: Int, y: Int): Boolean {
        return x < 0 || y < 0 || x >= gameSpec.boardSize || y >= gameSpec.boardSize
    }

    fun checkJumpConstraints(next: GameState,
                             x1: Int, x2: Int, y1: Int, y2: Int,
                             src: Int = at(x1, y1),
                             move: Move): Boolean {
        val dx = x2 - x1
        val dy = y2 - y1
        if (dx == 0 || dy == 0 || abs(dx) == abs(dy)) {
            // loops over squares in between location and destination
            for (i in 1 until max(abs(dx), abs(dy))) {

                // gets square you are inspecting
                val x3 = x1 + dx.sign * i
                val y3 = y1 + dy.sign * i
                val jumped = at(x3, y3)

                if (jumped == 0 && move.jump.none == DISALLOWED) {
                    return false
                } else if (jumped.sign == src.sign && move.jump.own == DISALLOWED) {
                    return false
                } else if (jumped.sign == -src.sign && move.jump.opponent == DISALLOWED) {
                    return false
                }

                if (jumped.sign == src.sign && move.jump.own == CAPTURE) {
                    next.setState(x3, y3, 0)
                } else if (jumped.sign == -src.sign && move.jump.opponent == CAPTURE) {
                    next.setState(x3, y3, 0)
                } else if (jumped.sign == -src.sign && move.jump.opponent == IMPRESS) {
                    next.setState(x3, y3, -jumped)
                }
            }
        }
        return true
    }


    private fun maybeAddMove(moves: MutableMap<Int, ArrayList<GameState>>,
                             move: GameGrammar.Move,
                             x1: Int, y1: Int, x2: Int, y2: Int,
                             src: Int = at(x1, y1)): Boolean {
        val next = GameState(this, x1, y1, x2, y2, src)
        val dst = at(x2, y2)

        if (!checkLandingConstraints(dst, src, move)) return false
        if (!checkJumpConstraints(next, x1, x2, y1, y2, src, move)) return false

        if (dst.sign != 0) {
            val action = if (dst.sign == src.sign) move.land.own else move.land.opponent
            when (action) {
                SWAP -> {
                    if (offBoard(x1, y1)) throw RuntimeException("illegal swap with virtual piece")
                    next.setState(x1, y1, dst)
                    next.setState(x2, y2, src)
                }
                CAPTURE -> {
                    next.setState(x2, y2, src)
                    if (!offBoard(x1, y1)) {
                        next.setState(x1, y1, 0)
                    }
                }
                STAY -> {
                    // src and dst pieces stay where they are
                }
                else -> throw RuntimeException("grammar doesn't specify disposition of non-empty destination")
            }
        } else {
            if (move.land.none == ALLOWED) {
                next.setState(x2, y2, src)
                if (!offBoard(x1, y1)) {
                    next.setState(x1, y1, 0)
                }
            } else if (move.land.none == DEPLOY) {
                // place a fresh piece from off board, leave src. where it is
                next.setState(x2, y2, src)
            } else {
                throw RuntimeException("grammar specifies swap or capture for empty cell")
            }
        }
        if (move.exchange.isNotEmpty()) {
            for (i in gameSpec.pieceList.indices) {
                if (gameSpec.pieceList[i].name == move.exchange) {
                    next.setState(x2, y2, if (whiteMove) i else -i)
                }
            }
        }
        if (move.`continue`) {
            next.whiteMove = whiteMove
        }
        val list = moves[move.priority]
        if (list == null) {
            moves[move.priority] = arrayListOf(next)
        } else {
            list.add(next)
        }
        return true
    }

    private fun forward(size: Int): List<Pair<Int, Int>> {
        return square(size).filter {
            if (gameSpec.boardSymmetry == NONE) {
                it.second > 0
            } else {
                if (whiteMove) it.second > 0
                else it.second < 0
            }
        }
    }

    private fun rank(row: Int): List<Pair<Int, Int>> {
        val adjY = (
                if (whiteMove || gameSpec.boardSymmetry == NONE) row - 1
                else gameSpec.boardSize - row)
        val moves = arrayListOf<Pair<Int, Int>>()
        for (x in 0 until gameSpec.boardSize) {
            moves.add(Pair(x, adjY))
        }
        return moves
    }

    private fun pass(): List<Pair<Int, Int>> {
        return arrayListOf(Pair(0, 0))
    }

    /**
     * Collects the moves that may be performed by an actual (or virtual) piece
     * A virtual piece would be one just off the board.
     *
     * @property moves the prioritized collection for accumulating moves
     * @property x the x position of the piece
     * @property y the y position of the piece (or -1 or boardSize for virtuals)
     * @property p the piece type id (must be provided if the piece is virtual)
     */
    private fun collectPieceMoves(moves: MutableMap<Int, ArrayList<GameState>>,
                                  x: Int, y: Int, p: Int = at(x, y)) {
        val piece = getPiece(p)
        for (move in piece.moveList) {
            val squares = arrayListOf<Pair<Int, Int>>()

            for (template in move.templateList) {
                val action = template.substring(0, 1)
                val (pattern, size_str) = template.substring(1).split("_")
                var size = size_str.toInt()
                if (size == 0) size = gameSpec.boardSize

                // convert offset to board position (result may be out of bounds)
                fun toBoard(offset: Pair<Int, Int>) = Pair(x + offset.first, y + offset.second)

                val newSquares = when (pattern) {
                    "square" -> square_m(size).map(::toBoard)
                    "plus" -> plus_m(size).map(::toBoard)
                    "cross" -> cross_m(size).map(::toBoard)
                    "forward" -> forward(size).map(::toBoard)
                    "pass" -> pass().map(::toBoard)
                    "rank" -> rank(size)
                    else -> arrayListOf()
                }

                when (action) {
                    "+" -> squares.addAll(newSquares)
                    "=" -> {
                        val intersection = squares.intersect(newSquares)
                        squares.clear()
                        squares.addAll(intersection)
                    }
                    else -> squares.removeAll(newSquares)
                }
            }
            for (square in squares) {
                val (x2, y2) = square
                if (x2 >= 0 && x2 < gameSpec.boardSize && y2 >= 0 && y2 < gameSpec.boardSize) {
                    maybeAddMove(moves, move, x, y, x2, y2, p)
                }
            }
        }
    }

    // for lazy init of nextMoves. should not be called directly
    fun getLegalNextStates(): ArrayList<GameState> {
        val states = TreeMap<Int, ArrayList<GameState>>()
        val playerSign = if (whiteMove) 1 else -1

        when (gameSpec.moveSource) {
            MoveSource.PIECES_ON_BOARD -> {
                gameBoard.forEachIndexed { y, row ->
                    row.forEachIndexed { x, pieceId ->
                        if (pieceId.sign == playerSign) {
                            collectPieceMoves(states, x, y)
                        }
                    }
                }
            }
            MoveSource.ENDS -> {
                for (x in 0 until gameSpec.boardSize) {
                    for (pieceId in gameSpec.pieceList.indices) {
                        collectPieceMoves(
                                states, x, -1, pieceId * playerSign)
                        collectPieceMoves(
                                states, x, gameSpec.boardSize, pieceId * playerSign)
                    }
                }
            }
            else -> {
                throw(RuntimeException("moveSource is neither on board or ends"))
            }
        }
        return states.lastEntry()?.component2() ?: ArrayList()
    }

    fun expand() {
        if (!leaf) {
            return
        }
        totalValue = when (outcome) {
            GameOutcome.UNDETERMINED -> {
                val (value, priors) = predict()
                for (move in nextMoves) {
                    move.prior = priors[move.getMoveIndex()]
                }
                value
            }
            GameOutcome.WIN_BLACK -> {
                if (whiteMove) -1f else 1f
            }
            GameOutcome.WIN_WHITE -> {
                if (whiteMove) 1f else -1f
            }
            GameOutcome.DRAW -> {
                0f
            }
        }
        visitCount = 1
        leaf = false
    }

    fun pieceCounts(): Pair<IntArray, IntArray> {
        val p1Counts = IntArray(gameSpec.pieceList.size, { 0 })
        val p2Counts = IntArray(gameSpec.pieceList.size, { 0 })
        for (row in gameBoard) {
            for (piece in row) {
                if (piece > 0) p1Counts[piece] += 1
                if (piece < 0) p2Counts[-piece] += 1
            }
        }
        return Pair(p1Counts, p2Counts)
    }

    fun maxSequenceLengths(n: Int): Pair<Int, Int> {
        val sz = gameSpec.boardSize
        fun atSafe(row: Int, col: Int) = if (row < 0 || col < 0 || row >= sz || col >= sz) 0 else at(row, col)


        var maxSeq = 0
        var minSeq = 0
        for (row in 0 until sz) {
            for (col in 0 until sz) {
                val counts = arrayOf(0, 0, 0, 0)
                for (i in 0 until n) {
                    counts[0] += atSafe(row, col + i).sign
                    counts[1] += atSafe(row + i, col).sign
                    counts[2] += atSafe(row + i, col + i).sign
                    counts[3] += atSafe(row + i, col - i).sign
                }
                maxSeq = max(maxSeq, counts.max() ?: 0)
                minSeq = min(minSeq, counts.min() ?: 0)
            }
        }
        return Pair(maxSeq, abs(minSeq))
    }

    fun outcomeByDecision(gameDecision: GameDecision): GameOutcome {
        return when (gameDecision) {
            GameDecision.WIN ->
                if (whiteMove) GameOutcome.WIN_WHITE else GameOutcome.WIN_BLACK
            GameDecision.LOSS ->
                if (whiteMove) GameOutcome.WIN_BLACK else GameOutcome.WIN_WHITE
            GameDecision.DRAW ->
                GameOutcome.DRAW
            GameDecision.COUNT_LIVE_PIECES -> {
                val (player1Count, player2Counts) = pieceCounts()
                if (player1Count.sum() == player2Counts.sum()) GameOutcome.DRAW
                else if (player1Count.sum() > player2Counts.sum()) GameOutcome.WIN_WHITE else GameOutcome.WIN_BLACK
            }
            GameDecision.COUNT_CAPTURED_PIECES -> {
                GameOutcome.UNDETERMINED // todo
            }
            else -> GameOutcome.UNDETERMINED

        }
    }

    // for lazy init of outcome. should not be called directly
    fun gameOutcome(): GameOutcome {
        val counts: Pair<IntArray, IntArray> by lazy {
            pieceCounts()
        }
        for (game_over in gameSpec.gameOverList) {
            when (game_over.condition) {
                Condition.NO_LEGAL_MOVE -> {
                    if (nextMoves.isEmpty()) {
                        return outcomeByDecision(game_over.decision)
                    }
                }
                Condition.MOVE_LIMIT -> {
                    if (moveDepth > game_over.param) {
                        return outcomeByDecision(game_over.decision)
                    }
                }
                Condition.BOARD_FULL -> {
                    if (numEmptyCells() == 0) {
                        return outcomeByDecision(game_over.decision)
                    }
                }
                Condition.KEY_PIECES_CAPTURED -> {
                    for (i in gameSpec.pieceList.indices) {
                        if (counts.first[i] < gameSpec.pieceList[i].min) {
                            return GameOutcome.WIN_BLACK
                        }
                        if (counts.second[i] < gameSpec.pieceList[i].min) {
                            return GameOutcome.WIN_WHITE
                        }
                    }
                }
                Condition.NO_PIECES_ON_BOARD -> {
                    if (counts.first.sum() == 0) {
                        return GameOutcome.WIN_BLACK
                    }
                    if (counts.second.sum() == 0) {
                        return GameOutcome.WIN_WHITE
                    }
                }
                Condition.N_IN_A_ROW -> {
                    val maxLengths = maxSequenceLengths(game_over.param)
                    if (maxLengths.first >= game_over.param) {
                        return GameOutcome.WIN_WHITE
                    }
                    if (maxLengths.second >= game_over.param) {
                        return GameOutcome.WIN_BLACK
                    }
                }
                else -> {
                }
            }
        }
        return GameOutcome.UNDETERMINED
    }


    /**
     * print statements for board
     *
     *  small board
     *  reg board
     *  large board
     *
     * **/


    private fun getRow(row: IntArray, i: Int) {
        val h_ = "\u001B[47m"
        val _h = "\u001B[0m"
        print("\u001B[40m    \u001B[0m")
        row.forEachIndexed { j, _ ->
            if ((i + j) % 2 == 0) print(h_)
            print("       $_h")
        }.also { println("\u001B[37m\u001B[40m    \u001B[0m") }
    }


    fun printBoard() {


        print("\u001B[40m        ")
        for (index in 0 until gameSpec.boardSize) print("       ")
        println("\u001B[0m")
        print("\u001B[40m        ")
        for (index in 0 until gameSpec.boardSize) print("       ")
        println("\u001B[0m")

        val h_ = "\u001B[47m"
        val _h = "\u001B[0m"
        val b_ = "\u001B[1m"

        gameBoard.reversed().forEachIndexed { i, row ->

            getRow(row, i)
            print("\u001B[40m    \u001B[0m")
            row.forEachIndexed { j, piece ->
                if ((i + j) % 2 == 0) print(h_)
                var pieceStr = gameSpec.pieceList[abs(piece)].name
                val symbol = if (piece < 0) "-" else if (piece > 0) " " else " ".also { pieceStr = " " }
                print("$b_  $symbol$pieceStr$symbol  $_h")

            }.also { println("\u001B[37m\u001B[40m  ${gameSpec.boardSize - i} \u001B[0m") }

            getRow(row, i)

        }
        print("\u001B[37m\u001B[40m    ")
        for (index in 0 until gameSpec.boardSize) print("       ")
        print("    \u001B[0m\n\u001B[37m\u001B[40m    ")
        for (index in 0 until gameSpec.boardSize) print("   " + ('a' + index) + "   ")
        print("    \u001B[0m")
        println("\n\n")

    }


}


fun main(args: Array<String>) {
    val game: String = if (args.size > 0) args[0] else readLine() ?: "chess"
    val specStr: String
    try {
        specStr = String(Files.readAllBytes(Paths.get("src/main/data/$game.textproto")))
    } catch (e: NoSuchFileException) {
        println("No game spec found for '${game}'.")
        return
    }
    val builder = GameSpec.newBuilder()
    TextFormat.getParser().merge(specStr, builder)
    val gameSpec = builder.apply {
        addPiece(0, builder.addPieceBuilder())
    }.build()


    if (args.size == 3) {
        tournament(gameSpec, args[1], args[2])
    } else {
        val modelFile: String? = if (args.size == 2) args[1] else null
        val outputStream = FileOutputStream("${gameSpec.name}.${System.currentTimeMillis()}")
        while (true) play(gameSpec, outputStream, modelFile)
    }
}


