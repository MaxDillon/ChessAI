package maximum.industries

import maximum.industries.GameGrammar.*
import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

enum class Outcome { UNDETERMINED, WIN, LOSE, DRAW }
enum class Player { WHITE, BLACK }

//=================================================================================
// Template generating functions (memoized outside GameState for best reuse)
//=================================================================================
class Memoize<in T, out R>(val f: (T) -> R) : (T) -> R {
    private val cache = mutableMapOf<T, R>()
    override fun invoke(x: T): R = cache.getOrPut(x) { f(x) }
}

val square = Memoize { size: Int ->
    val moves = arrayListOf<Pair<Int, Int>>()
    for (i in 0..size) {
        for (j in 1..size) {
            moves.add(Pair(i, j))
            moves.add(Pair(-j, i))
            moves.add(Pair(-i, -j))
            moves.add(Pair(j, -i))
        }
    }
    moves
}

val plus = Memoize { size: Int ->
    square(size).filter { it.first == 0 || it.second == 0 }
}

val cross = Memoize { size: Int ->
    square(size).filter { Math.abs(it.first) == Math.abs(it.second) }
}

val forward = Memoize { sizeAndSign: Pair<Int, Int> ->
    square(sizeAndSign.first).filter { it.second * sizeAndSign.second > 0 }
}

val rank = Memoize { sizeAndRow: Pair<Int, Int> ->
    List(sizeAndRow.first) { Pair(it, sizeAndRow.second) }
}

// pass is a trivial zero-offset move
private fun pass(): List<Pair<Int, Int>> {
    return arrayListOf(Pair(0, 0))
}
//=================================================================================

class GameState {
    val gameSpec: GameSpec
    private val gameBoard: Array<IntArray>
    val player: Player
    val p1: Int // the piece that was moved (needed to construct input channels)
    val x1: Int // p1's src x
    val y1: Int // p1's src y
    val x2: Int // p1's dst x
    val y2: Int // p1's dst y
    val moveDepth: Int
    val history: ArrayList<Int>
    val nextMoves: ArrayList<GameState> by lazy {
        getNextStates()
    }
    val outcome: Outcome by lazy {
        gameOutcome()
    }

    constructor(gameSpec: GameSpec) {
        this.gameSpec = gameSpec
        gameBoard = Array(gameSpec.boardSize) { IntArray(gameSpec.boardSize) { 0 } }
        gameSpec.pieceList.forEachIndexed { pieceType, piece ->
            piece.placementList.forEach { placement ->
                val (x, y) = placement.substring(1).split("y").map { it.toInt() - 1 }
                if (gameBoard[y][x] == 0) {
                    gameBoard[y][x] = pieceType
                    val oppositeX = if (gameSpec.boardSymmetry == Symmetry.ROTATE)
                        gameSpec.boardSize - 1 - x else x
                    val oppositeY = gameSpec.boardSize - 1 - y
                    gameBoard[oppositeY][oppositeX] = -pieceType
                } else throw RuntimeException("you can't place a piece at $x,$y")
            }
        }
        this.player = Player.WHITE
        this.p1 = 0
        this.x1 = -1
        this.y1 = -1
        this.x2 = -1
        this.y2 = -1
        this.moveDepth = 0
        this.history = arrayListOf(gameBoard.contentDeepHashCode())
    }

    constructor(gameSpec: GameSpec, gameBoard: Array<IntArray>, player: Player,
                p1: Int, x1: Int, y1: Int, x2: Int, y2: Int, moveDepth: Int,
                history: ArrayList<Int> = ArrayList()) {
        this.gameSpec = gameSpec
        this.gameBoard = gameBoard
        this.player = player
        this.p1 = p1
        this.x1 = x1
        this.y1 = y1
        this.x2 = x2
        this.y2 = y2
        this.moveDepth = moveDepth
        this.history = history
        this.history.add(gameBoard.contentDeepHashCode())
    }

    override fun toString() =
            if (x2 < 0 && y2 < 0) {
                "START"
            } else {
                val status = if (outcome == Outcome.UNDETERMINED) "" else " $outcome"
                "$moveDepth: ${'a' + x1}${y1 + 1} -> ${'a' + x2}${y2 + 1}$status"
            }

    // TODO: consider changing everything to atRowCol instead of X,Y, which are consistently confusing
    fun at(x: Int, y: Int): Int {
        return gameBoard[y][x]
    }

    fun winFor(parent: GameState): Boolean {
        return if (player.eq(parent.player)) outcome == Outcome.WIN else outcome == Outcome.LOSE
    }

    fun lossFor(parent: GameState): Boolean {
        return if (player.eq(parent.player)) outcome == Outcome.LOSE else outcome == Outcome.WIN
    }

    //=================================================================================
    // Functions for constructing successor states
    //=================================================================================

    private fun getPieceDefinition(pieceId: Int): Piece {
        return gameSpec.getPiece(abs(pieceId))
    }

    private fun offBoard(x: Int, y: Int): Boolean {
        return x < 0 || y < 0 || x >= gameSpec.boardSize || y >= gameSpec.boardSize
    }

    private fun checkLandingConstraints(x2: Int, y2: Int,
                                        srcPiece: Int, move: Move): Boolean {
        val dstPiece = at(x2, y2)
        if (dstPiece == 0 && move.land.none == GameGrammar.Outcome.DISALLOWED) {
            return false
        } else if (dstPiece.sign == srcPiece.sign && move.land.own == GameGrammar.Outcome.DISALLOWED) {
            return false
        } else if (dstPiece.sign == -srcPiece.sign && move.land.opponent == GameGrammar.Outcome.DISALLOWED) {
            return false
        }
        return true
    }

    private fun iterateJumped(x1: Int, y1: Int, x2: Int, y2: Int,
                              visitor: (x: Int, y: Int) -> Boolean): Boolean {
        val dx = x2 - x1
        val dy = y2 - y1
        if (dx == 0 || dy == 0 || abs(dx) == abs(dy)) {
            for (i in 1 until max(abs(dx), abs(dy))) {
                val x3 = x1 + dx.sign * i
                val y3 = y1 + dy.sign * i
                if (!visitor(x3, y3)) return false
            }
        }
        return true
    }

    private fun checkJumpingConstraints(x1: Int, y1: Int, x2: Int, y2: Int,
                                        srcPiece: Int, move: Move): Boolean {
        return iterateJumped(x1, y1, x2, y2) { x3, y3 ->
            val jumpedPiece = at(x3, y3)
            if (jumpedPiece == 0 && move.jump.none == GameGrammar.Outcome.DISALLOWED) {
                false
            } else if (jumpedPiece.sign == srcPiece.sign && move.jump.own == GameGrammar.Outcome.DISALLOWED) {
                false
            } else if (jumpedPiece.sign == -srcPiece.sign && move.jump.opponent == GameGrammar.Outcome.DISALLOWED) {
                false
            } else {
                true
            }
        }
    }

    // Creates a GameState for a given move. Assumes all error checking has already been done and
    // the specified move is valid.
    private fun createAndAddState(nextStates: MutableMap<Int, ArrayList<GameState>>,
                                  x1: Int, y1: Int,
                                  x2: Int, y2: Int,
                                  srcPiece: Int,
                                  move: Move) {
        val nextBoard = Array(gameBoard.size) { gameBoard[it] }
        fun set(x: Int, y: Int, p: Int) {
            // note: the == array comparison is intended here to support copy-on-write
            if (gameBoard[y] == nextBoard[y]) nextBoard[y] = gameBoard[y].clone()
            nextBoard[y][x] = p
        }

        val dstPiece = at(x2, y2)

        // first apply any board updates for jumped squares
        iterateJumped(x1, y1, x2, y2) { x3, y3 ->
            val jumpedPiece = at(x3, y3)
            if (jumpedPiece.sign == srcPiece.sign && move.jump.own == GameGrammar.Outcome.CAPTURE) {
                set(x3, y3, 0)
            } else if (jumpedPiece.sign == -srcPiece.sign && move.jump.opponent == GameGrammar.Outcome.CAPTURE) {
                set(x3, y3, 0)
            } else if (jumpedPiece.sign == -srcPiece.sign && move.jump.opponent == GameGrammar.Outcome.IMPRESS) {
                set(x3, y3, -jumpedPiece)
            }
            true
        }
        // then apply changes to the src and dst locations
        if (dstPiece.sign != 0) {
            val action = if (dstPiece.sign == srcPiece.sign) move.land.own else move.land.opponent
            when (action) {
                GameGrammar.Outcome.SWAP -> {
                    if (offBoard(x1, y1)) throw RuntimeException("illegal swap with virtual piece")
                    set(x1, y1, dstPiece)
                    set(x2, y2, srcPiece)
                }
                GameGrammar.Outcome.CAPTURE -> {
                    set(x2, y2, srcPiece)
                    if (!offBoard(x1, y1)) {
                        set(x1, y1, 0)
                    }
                }
                GameGrammar.Outcome.STAY -> {
                    // src and dst pieces stay where they are
                }
                else -> throw RuntimeException("grammar doesn't specify disposition of non-empty destination")
            }
        } else {
            if (move.land.none == GameGrammar.Outcome.ALLOWED) {
                set(x2, y2, srcPiece)
                if (!offBoard(x1, y1)) {
                    set(x1, y1, 0)
                }
            } else if (move.land.none == GameGrammar.Outcome.DEPLOY) {
                // place a fresh piece from off board, leave src. where it is
                set(x2, y2, srcPiece)
            } else {
                throw RuntimeException("grammar specifies swap or capture for empty cell")
            }
        }
        // then perform any post-move piece exchange
        if (move.exchange.isNotEmpty()) {
            for (i in gameSpec.pieceList.indices) {
                if (gameSpec.pieceList[i].name == move.exchange) {
                    set(x2, y2, if (player.eq(Player.WHITE)) i else -i)
                }
            }
        }
        // then create the new GameState and add to the appropriate prioritized list
        val nextPlayer = if (move.`continue`) {
            player
        } else {
            if (player.eq(Player.WHITE)) Player.BLACK else Player.WHITE
        }
        val nextState = GameState(gameSpec, nextBoard, nextPlayer, srcPiece, x1, y1, x2, y2,
                                  moveDepth + 1, ArrayList(history))
        nextStates.getOrPut(move.priority) { ArrayList() }.add(nextState)
    }

    /**
     * Collects the nextStates that may be performed by an actual (or virtual) piece
     * A virtual piece would be one just off the board.
     *
     * @property nextStates the prioritized collection for accumulating nextStates
     * @property x1 the x position of the piece
     * @property y1 the y position of the piece (or -1 or boardSize for virtuals)
     * @property srcPiece the piece type id (must be provided if the piece is virtual)
     */
    private fun collectNextStates(nextStates: MutableMap<Int, ArrayList<GameState>>,
                                  x1: Int, y1: Int, srcPiece: Int = at(x1, y1)) {
        val piece = getPieceDefinition(srcPiece)
        val forwardSign: Int =
                if (gameSpec.boardSymmetry == Symmetry.NONE || player.eq(Player.WHITE)) 1 else -1

        for (move in piece.moveList) {
            val targetSquares = arrayListOf<Pair<Int, Int>>()

            for (template in move.templateList) {
                val action = template.substring(0, 1)
                val (pattern, size_str) = template.substring(1).split("_")
                var size = size_str.toInt()
                if (size == 0) size = gameSpec.boardSize

                // convert offset to board position (result may be out of bounds)
                fun toBoard(offset: Pair<Int, Int>) = Pair(x1 + offset.first, y1 + offset.second)

                val newSquares = when (pattern) {
                    "square" -> square(size).map(::toBoard)
                    "plus" -> plus(size).map(::toBoard)
                    "cross" -> cross(size).map(::toBoard)
                    "forward" -> forward(Pair(size, forwardSign)).map(::toBoard)
                    "backward" -> forward(Pair(size, -forwardSign)).map(::toBoard)
                    "rank" -> {
                        val row = if (player.eq(Player.WHITE) || gameSpec.boardSymmetry == Symmetry.NONE) {
                            size - 1
                        } else {
                            gameSpec.boardSize - size
                        }
                        rank(Pair(gameSpec.boardSize, row))
                    } // we do NOT map rank() to board
                    "pass" -> pass().map(::toBoard)
                    else -> arrayListOf()
                }

                when (action) {
                    "+" -> targetSquares.addAll(newSquares)
                    "=" -> {
                        val intersection = targetSquares.intersect(newSquares)
                        targetSquares.clear()
                        targetSquares.addAll(intersection)
                    }
                    else -> targetSquares.removeAll(newSquares)
                }
            }
            for (square in targetSquares) {
                val (x2, y2) = square
                if (x2 < 0 || x2 >= gameSpec.boardSize) continue
                if (y2 < 0 || y2 >= gameSpec.boardSize) continue
                if (!checkLandingConstraints(x2, y2, srcPiece, move)) continue
                if (!checkJumpingConstraints(x1, y1, x2, y2, srcPiece, move)) continue
                createAndAddState(nextStates, x1, y1, x2, y2, srcPiece, move)
            }
        }
    }

    // Note: getNextStates() is for lazy init of nextMoves. should not be called directly
    private fun getNextStates(): ArrayList<GameState> {
        val states = TreeMap<Int, ArrayList<GameState>>() // map of priorities => lists of states
        val playerSign = if (player.eq(Player.WHITE)) 1 else -1

        when (gameSpec.moveSource) {
            MoveSource.PIECES_ON_BOARD -> {
                gameBoard.forEachIndexed { y, row ->
                    row.forEachIndexed { x, pieceId ->
                        if (pieceId.sign == playerSign) {
                            collectNextStates(states, x, y)
                        }
                    }
                }
            }
            MoveSource.ENDS -> {
                for (x in 0 until gameSpec.boardSize) {
                    for (pieceId in gameSpec.pieceList.indices) {
                        collectNextStates(
                                states, x, -1, pieceId * playerSign)
                        collectNextStates(
                                states, x, gameSpec.boardSize, pieceId * playerSign)
                    }
                }
            }
            else -> throw(RuntimeException("moveSource is neither on board or ends"))
        }
        // return moves in highest priority class or an empty list if no legal moves
        return states.lastEntry()?.component2() ?: ArrayList()
    }

    //=================================================================================
    // Functions for evaluating end-of-game conditions
    //=================================================================================

    private fun numEmptyCells(): Int {
        return gameBoard.map { it.filter { it == 0 }.count() }.sum()
    }

    private fun pieceCounts(): Pair<IntArray, IntArray> {
        val whiteCounts = IntArray(gameSpec.pieceList.size) { 0 }
        val blackCounts = IntArray(gameSpec.pieceList.size) { 0 }
        for (row in gameBoard) {
            for (piece in row) {
                if (piece > 0) whiteCounts[piece] += 1
                if (piece < 0) blackCounts[-piece] += 1
            }
        }
        return Pair(whiteCounts, blackCounts)
    }

    private fun maxSequenceLengths(n: Int): Pair<Int, Int> {
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
        val longestWhite = maxSeq
        val longestBlack = abs(minSeq)
        return Pair(longestWhite, longestBlack)
    }

    private fun outcomeByDecision(gameDecision: GameDecision): Outcome {
        return when (gameDecision) {
            GameDecision.WIN -> Outcome.WIN
            GameDecision.LOSS -> Outcome.LOSE
            GameDecision.DRAW -> Outcome.DRAW
            GameDecision.COUNT_LIVE_PIECES -> {
                val (whiteCount, blackCount) = pieceCounts()
                if (whiteCount.sum() == blackCount.sum())
                    Outcome.DRAW
                else if (whiteCount.sum() > blackCount.sum()) {
                    if (player.eq(Player.WHITE)) Outcome.WIN else Outcome.LOSE
                } else {
                    if (player.eq(Player.WHITE)) Outcome.LOSE else Outcome.WIN
                }
            }
            GameDecision.COUNT_CAPTURED_PIECES -> {
                Outcome.UNDETERMINED // TODO: implement
            }
            else -> Outcome.UNDETERMINED

        }
    }

    // for lazy init of outcome. should not be called directly
    private fun gameOutcome(): Outcome {
        val (whiteCounts, blackCounts) = pieceCounts()
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
                        if (whiteCounts[i] < gameSpec.pieceList[i].min) {
                            return if (player.eq(Player.WHITE)) Outcome.LOSE else Outcome.WIN
                        }
                        if (blackCounts[i] < gameSpec.pieceList[i].min) {
                            return if (player.eq(Player.WHITE)) Outcome.WIN else Outcome.LOSE
                        }
                    }
                }
                Condition.NO_PIECES_ON_BOARD -> {
                    if (whiteCounts.sum() == 0) {
                        return if (player.eq(Player.WHITE)) Outcome.LOSE else Outcome.WIN
                    }
                    if (blackCounts.sum() == 0) {
                        return if (player.eq(Player.WHITE)) Outcome.WIN else Outcome.LOSE
                    }
                }
                Condition.N_IN_A_ROW -> {
                    val (longestWhite, longestBlack) = maxSequenceLengths(game_over.param)
                    if (longestWhite >= game_over.param) {
                        return if (player.eq(Player.WHITE)) Outcome.WIN else Outcome.LOSE
                    }
                    if (longestBlack >= game_over.param) {
                        return if (player.eq(Player.WHITE)) Outcome.LOSE else Outcome.WIN
                    }
                }
                Condition.REPEATED_POSITION -> {
                    var count = 0
                    for (position in history) {
                        if (position == history.last()) count++
                    }
                    if (count == game_over.param) return Outcome.DRAW
                }
                else -> {
                }
            }
        }
        return Outcome.UNDETERMINED
    }

    //=================================================================================
    // Functions for pretty printing board states
    //=================================================================================

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
        print("  ${if (player == Player.WHITE) "W" else "B"} \u001B[0m")
        println("\n")
    }
}
