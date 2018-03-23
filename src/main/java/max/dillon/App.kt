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

        newStates.add(next)
        return true
    }

    fun getPieceMoves(x: Int, y: Int): ArrayList<GameState> {
        val newStates = arrayListOf<GameState>()

        val piece = getPiece(at(x, y)) // gets current piece in position x,y
        piece.moveList.forEach {
            val squares = arrayListOf<Pair<Int, Int>>()
            it.templateList.forEach {
                val sign = it.substring(0, 1)
                val (pattern, size_str) = it.substring(1).split("_")
                var size = size_str.toInt()
                if (size == 0) size = gameSpec.boardSize
                val newSquares = when (pattern) {
                    "square" -> square(size)
                    "plus" -> plus(size)
                    "cross" -> cross(size)
                    "forward" -> forward(size, whiteMove)
                    else -> arrayListOf()
                }

                if (sign == "+") {
                    squares.addAll(newSquares)
                } else {
                    squares.removeAll(newSquares)
                }
            }
            for (square in squares) {
                // have valid move offsets ignoring board boundaries and jump and landing constraints.
                val x2 = x + square.first
                val y2 = y + square.second
                if (x2 >= 0 && x2 < gameSpec.boardSize && y2 >= 0 && y2 < gameSpec.boardSize) {
                    maybeAddMove(newStates, it, x, y, x2, y2)
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

}


fun printArray(array: Array<IntArray>, grammar: GameSpec) {
    println("# # # # # # # # # # # # # # # # # #")
    println("# --- --- --- --- --- --- --- --- #")
    array.forEach {
        print("#|")
        it.forEach {
            val sign1 = if (it<1) "[" else " "
            val sign2 = if (it<1) "]" else " "

            val name = sign1+grammar.getPiece( abs(it) ).name+sign2


//            if (it >= 0) print(" ")
            print("${if (it == 0) "   " else name}|")
        }
        println("#\n# --- --- --- --- --- --- --- --- #")
    }
    println("# # # # # # # # # # # # # # # # # #\n\n")
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

fun forward(size: Int, white: Boolean): List<Pair<Int, Int>> {
    return square(size).filter { white && it.second > 0 || !white && it.second < 0 }
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
        printArray(state.gameBoard, gameSpec)
        val nextStates = state.getLegalNextStates()
        if (nextStates.size == 0) return
        state = nextStates[rand.nextInt(nextStates.size)]
    }
}


