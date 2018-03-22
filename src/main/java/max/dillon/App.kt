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


class GameState {
    var gameBoard: Array<IntArray>
    private val gameSpec: GameSpec
    private var whiteMove: Boolean = true

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
                    gameBoard[oppositeY][oppositeX] = - pieceType
                } else throw RuntimeException("you cant place a piece at $x,$y")
            }
        }
    }







    private fun at(x: Int, y: Int): Int {
        printArray(gameBoard)
        println(gameBoard[x][y])
        println("$x, $y")
        return gameBoard[x][y]
    }

    constructor(gameSpec: GameSpec, gameBoard: Array<IntArray>, whiteMove: Boolean) : this(gameSpec) {
        this.gameBoard = gameBoard
        this.whiteMove = whiteMove
    }

    fun clone(): GameState {

        val array = Array(gameBoard.size) { gameBoard[it].clone() }
        return GameState(gameSpec, array, whiteMove)

    }

    private fun getPiece(i: Int): GameGrammar.Piece = gameSpec.getPiece(i)

    private fun setState(x: Int, y: Int, state: Int) {
        gameBoard[x][y] = state
    }

    fun checkSquare(move: GameGrammar.Move, x1: Int, y1: Int, x2: Int, y2: Int): Boolean {

        val d1 = x2 - x1
        val d2 = y2 - y1

        if (d1 == 0 || d2 == 0 || abs(d1) == abs(d2)) { // if we have a plus or cross
            val steps = max(abs(d1), abs(d2)) - 1

            for (i in 0 until steps) {
                val (otherX, otherY) = Pair(x1 + d1.sign * i, y1 + d2.sign * i)
                val otherPiece = at(otherX, otherY)
                if (otherPiece == 0 && move.jump.none == DISALLOWED) return false
                if (otherPiece > 0 && move.jump.own == DISALLOWED) return false
                if (otherPiece < 0 && move.jump.opponent == DISALLOWED) return false
                if (otherPiece < 0 && move.jump.opponent == CAPTURE) setState(otherX, otherY, 0); return false

            }
            val destPiece = at(x2, y2)
            if (destPiece == 0 && move.land.none == DISALLOWED) return false
            if (destPiece > 0 && move.land.own == DISALLOWED) return false
            if (destPiece < 0 && move.land.opponent == DISALLOWED) return false

        }
        setState(x2, y2, at(x1, y1))
        setState(x1, y1, 0)
        return true
    }


    fun getPieceMoves(x: Int, y: Int): ArrayList<Pair<Int,Int>> {


        val finalSquares = arrayListOf<Pair<Int, Int>>()

        val piece = getPiece(at(x, y)) // gets current piece in position x,y
        piece.moveList.forEach {
            val squares = arrayListOf<Pair<Int, Int>>()

            it.templateList.forEach {
                val sign = it.substring(0, 1)
                val (pattern, size_str) = it.substring(1).split("_")
                var size = size_str.toInt()
                if (size == 0) size = gameSpec.boardSize
                println(pattern)
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
            // have valid move offsets ignoring board boundaries and jump and landing constraints.
            val iterator = squares.listIterator()
            while (iterator.hasNext()) {
                val offset = iterator.next()
                val cell = Pair(x + offset.first, y + offset.second)
                if (cell.first < 0 || cell.first >= gameSpec.boardSize ||
                        cell.second < 0 || cell.second >= gameSpec.boardSize) {
                    iterator.remove()
                } else {
                    iterator.set(cell)
                }

            }
            // have valid positions on the board

            // apply jump and landing constraints.

            // then create mutated board states (apply the move and any captures or exchanges)
            finalSquares += squares
        }
        return finalSquares

    }

}


fun printArray(anArray: Array<IntArray>) {
    anArray.forEach {
        it.forEach {
            if (it >= 0) print(" ")
            print("$it  ")
        }
        println("\n")
    }
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


fun getLegalNextStates(state: GameState): ArrayList<Pair<Int,Int>> {
    val states = arrayListOf<Pair<Int,Int>>()
    state.gameBoard.forEachIndexed { x, row ->
        row.forEachIndexed { y, piece ->
            if (piece != 0) states.addAll(state.getPieceMoves(x, y))
        }
    }
    return states
}

fun main(args: Array<String>) {
    val str = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))
    val builder = GameSpec.newBuilder()
    TextFormat.getParser().merge(str, builder)
    val gameSpec = builder.build()

    val state = GameState(gameSpec)


    println(state.getPieceMoves(0,3))
}


