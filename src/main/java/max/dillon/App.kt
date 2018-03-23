package max.dillon

import com.google.protobuf.TextFormat
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign

import max.dillon.GameGrammar.Symmetry.*
import max.dillon.GameGrammar.Outcome.*
import max.dillon.GameGrammar.*
import java.util.*

class GameState {
    var gameBoard: Array<IntArray>
    private val gameSpec: GameSpec
    var whiteMove: Boolean = true

    constructor(gameSpec: GameSpec) {
        this.gameSpec = gameSpec
        gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })
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

    private fun at(x: Int, y: Int): Int {
        return gameBoard[y][x]
    }

    constructor(gameSpec: GameSpec, gameBoard: Array<IntArray>, whiteMove: Boolean) : this(gameSpec) {
        this.gameBoard = gameBoard
        this.whiteMove = whiteMove
    }

    fun initNext(): GameState {
        val array = Array(gameBoard.size) { gameBoard[it].clone() }
        return GameState(gameSpec, array, !whiteMove)
    }

    private fun getPiece(i: Int): GameGrammar.Piece = gameSpec.getPiece(abs(i))

    private fun setState(x: Int, y: Int, state: Int) {
        gameBoard[y][x] = state
    }

    fun maybeAddMove(newStates: ArrayList<GameState>, move: GameGrammar.Move,
                     x1: Int, y1: Int, x2: Int, y2: Int): Boolean {
        val next = initNext()
        val src = at(x1, y1)
        val dst = at(x2, y2)
        // check legality of landing constraints
        if (dst == 0 && move.land.none == DISALLOWED) {
            return false
        } else if (dst.sign == src.sign && move.land.own == DISALLOWED) {
            return false
        } else if (dst.sign == -src.sign && move.land.opponent == DISALLOWED) {
            return false
        }
        // check legality of jump constraints
        val dx = x2 - x1
        val dy = y2 - y1
        if (dx == 0 || dy == 0 || abs(dx) == abs(dy)) {
            for (i in 1 until max(abs(dx), abs(dy))) {
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
                }
            }
        }

        if (dst.sign != 0) {
            val action = if (dst.sign == src.sign) move.land.own else move.land.opponent
            if (action == SWAP) {
                next.setState(x1, y1, dst)
                next.setState(x2, y2, src)
            } else if (action == CAPTURE) {
                next.setState(x2, y2, src)
                next.setState(x1, y1, 0)
            } else {
                throw RuntimeException("grammar doesn't specify disposition of non-empty destination")
            }
        } else {
            if (move.land.none == ALLOWED) {
                next.setState(x2, y2, src)
                next.setState(x1, y1, 0)
            } else {
                throw RuntimeException("grammar specifies swap or capture for empty cell")
            }
        }
        if (move.promoteCount == 0) {
            newStates.add(next)
        } else {
            for (name in move.promoteList) {
                val promo = next.initNext()
                promo.whiteMove = next.whiteMove
                for (i in gameSpec.pieceList.indices) {
                    if (gameSpec.pieceList[i].name == name) {
                        promo.setState(x2, y2, if (whiteMove) i else -1)
                    }
                }
                newStates.add(promo)
            }
        }
        return true
    }

    fun square(size: Int): List<Pair<Int, Int>> {
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

    fun plus(size: Int): List<Pair<Int, Int>> {
        return square(size).filter { it.first == 0 || it.second == 0 }
    }

    fun cross(size: Int): List<Pair<Int, Int>> {
        return square(size).filter { Math.abs(it.first) == Math.abs(it.second) }
    }

    fun forward(size: Int): List<Pair<Int, Int>> {
        return square(size).filter { whiteMove && it.second > 0 || !whiteMove && it.second < 0 }
    }

    fun rank(row: Int): List<Pair<Int, Int>> {
        val adjRow = if (whiteMove) row else gameSpec.boardSize - 1 - row
        val moves = arrayListOf<Pair<Int, Int>>()
        for (col in 0 until gameSpec.boardSize) {
            moves.add(Pair(adjRow, col))
        }
        return moves
    }

    fun getPieceMoves(x: Int, y: Int): ArrayList<GameState> {
        val newStates = arrayListOf<GameState>()
        val piece = getPiece(at(x, y)) // gets current piece in position x,y
        for (move in piece.moveList) {
            var squares = arrayListOf<Pair<Int, Int>>()
            for (template in move.templateList) {
                val action = template.substring(0, 1)
                val (pattern, size_str) = template.substring(1).split("_")
                var size = size_str.toInt()
                if (size == 0) size = gameSpec.boardSize
                fun toBoard(offset: Pair<Int, Int>) = Pair(x + offset.first, y + offset.second)
                val newSquares = when (pattern) {
                    "square" -> square(size).map(::toBoard)
                    "plus" -> plus(size).map(::toBoard)
                    "cross" -> cross(size).map(::toBoard)
                    "forward" -> forward(size).map(::toBoard)
                    "rank" -> rank(size)
                    else -> arrayListOf()
                }
                if (action == "+") {
                    squares.addAll(newSquares)
                } else if (action == "=") {
                    val intersection = squares.intersect(newSquares)
                    squares.clear()
                    squares.addAll(intersection)
                } else {
                    squares.removeAll(newSquares)
                }
            }
            for (square in squares) {
                val (x2, y2) = square
                if (x2 >= 0 && x2 < gameSpec.boardSize && y2 >= 0 && y2 < gameSpec.boardSize) {
                    maybeAddMove(newStates, move, x, y, x2, y2)
                }
            }
        }
        return newStates
    }

    fun getLegalNextStates(): ArrayList<GameState> {
        val states = arrayListOf<GameState>()
        gameBoard.forEachIndexed { y, row ->
            row.forEachIndexed { x, piece ->
                if (piece.sign == if (whiteMove) 1 else -1) states.addAll(getPieceMoves(x, y))
            }
        }
        return states
    }

    fun gameOver(): Boolean {
        val p1Counts = IntArray(gameSpec.pieceList.size, { 0 })
        val p2Counts = IntArray(gameSpec.pieceList.size, { 0 })
        for (row in gameBoard) {
            for (piece in row) {
                if (piece > 0) p1Counts[piece] += 1
                if (piece < 0) p2Counts[-piece] += 1
            }
        }
        for (i in gameSpec.pieceList.indices) {
            if (p1Counts[i] < gameSpec.pieceList[i].min ||
                    p2Counts[i] < gameSpec.pieceList[i].min) {
                p1Counts.forEach { print(" $it") }.also { println() }
                p2Counts.forEach { print(" $it") }.also { println() }
                println()
                return true
            }
        }
        return false
    }

    fun printBoard() {
        println("# # # # # # # # # # # # # # # # # #")
        println("# ------------------------------- #")
        gameBoard.forEach {
            print("#|")
            it.forEach {
                if (it >= 0) print(" ")
                print("${if (it == 0) " " else it.toString()} |")
            }
            println("#\n# --- --- --- --- --- --- --- --- #")
        }
        println("# # # # # # # # # # # # # # # # # #\n\n")
    }
}

fun main(args: Array<String>) {
    val str = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))
    val builder = GameSpec.newBuilder()
    TextFormat.getParser().merge(str, builder)

    val gameSpec = builder.apply {
        addPiece(0, builder.addPieceBuilder())
    }.build()

    val rand = Random()
    var state = GameState(gameSpec)
    
    while (true) {
        state.printBoard()
        if (state.gameOver()) break
        val nextStates = state.getLegalNextStates()
        if (nextStates.size == 0) return
        state = nextStates[rand.nextInt(nextStates.size)]
    }
}


