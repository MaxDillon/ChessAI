package max.dillon

import com.google.protobuf.TextFormat
import java.nio.file.Files
import java.nio.file.Paths


class GameState(val gameSpec: GameGrammar.GameSpec) {
    var gameBoard: Array<Array<Int>>
    var whiteMove: Boolean = true

    init {
        gameBoard = Array(gameSpec.boardSize, { Array(gameSpec.boardSize, { 0 }) })
        gameSpec.pieceList.forEachIndexed { index, piece ->
            val pieceType = index + 1
            piece.placementList.forEach { placement ->
                val (x, y) = placement.substring(1).split("y").map { it.toInt() - 1 }
                println("$x,$y")
                if ((gameBoard[y][x] == 0)) {
                    gameBoard[y][x] = pieceType
                    var oppositeX = x
                    if (gameSpec.boardSymmetry == GameGrammar.Symmetry.ROTATE) {
                        oppositeX = gameSpec.boardSize - 1 - x
                    }
                    gameBoard[gameSpec.boardSize - 1 - y][oppositeX] = -pieceType
                } else throw RuntimeException("you cant place a piece at $x,$y")
            }
        }
    }


    fun pieceAt(x: Int, y: Int): GameGrammar.Piece {
        return gameSpec.getPiece(gameBoard[x][y])
    }


    constructor(gameSpec: GameGrammar.GameSpec, gameBoard: Array<Array<Int>>, whiteMove: Boolean) : this(gameSpec) {
        this.gameBoard = gameBoard
        this.whiteMove = whiteMove
    }


    fun getPieceMoves( x: Int, y: Int ): ArrayList<GameState> {

        val newStates = arrayListOf<GameState>()
        val piece = pieceAt(x,y) // gets current piece in position x,y
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
                    else -> arrayListOf<Pair<Int, Int>>()
                }

                if (sign == "+") {
                    squares.addAll(newSquares)
                } else {
                    squares.removeAll(newSquares)
                }
            }
            // have valid move offsets ignoring board boundaries and jump and landing constraints.
            var iterator = squares.listIterator()
            while (iterator.hasNext()) {
                var offset = iterator.next()
                var cell = Pair(x + offset.first, y + offset.second)
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
        }
        TODO("finish")
        return newStates
    }


}








fun printArray(anArray: Array<Array<Int>>) {
    anArray.forEach {
        it.forEach {
            if (it >= 0) print(" ")
            print("$it  ")
        }
        println("\n")
    }
}

fun square(size: Int): List<Pair<Int, Int>> {
    var moves = arrayListOf<Pair<Int, Int>>()
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


fun getLegalNextStates(state: GameState): ArrayList<GameState> {
    val states = arrayListOf<GameState>()
    state.gameBoard.forEachIndexed { x, row ->
        row.forEachIndexed { y, piece ->
            if (piece != 0) states.addAll(state.getPieceMoves( x, y))
        }
    }
    return states
}

fun main(args: Array<String>) {
    val str = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))
    val builder = GameGrammar.GameSpec.newBuilder()
    TextFormat.getParser().merge(str, builder)
    val gameSpec = builder.build()

    println(gameSpec.name)
    println(gameSpec.boardSize)

    val state = GameState(gameSpec)
    // play game

    println("\nboard:\n")

    printArray(state.gameBoard)
    //getBoardStates(game, gameBoard)
}


