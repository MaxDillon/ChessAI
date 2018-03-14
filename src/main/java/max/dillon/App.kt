package max.dillon


import java.util.Arrays
import com.google.protobuf.TextFormat
import sun.plugin.dom.exception.InvalidStateException
import java.nio.file.Files
import java.nio.file.Paths


fun printArray(anArray: Array<Array<Int>>) {
    anArray.forEach {
        it.forEach {
            val c = it.toString()
            if (it>=0) print(" ")
            print(c)
            print("  ")
        }
        println("\n")
    }
}

fun getPieceMoves(x: Int, y: Int): Int {


    return "".hashCode()
}


fun getBoardStates(board: Array<Array<Int>>) {
    val possibleStates = arrayListOf<Int>()
    board.forEachIndexed { x, list->
        list.forEachIndexed { y, piece ->
            if(piece != 0) possibleStates.add(getPieceMoves(x,y))

        }
    }


}


fun main(args: Array<String>) {
    val str = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))

    val builder = GameGrammar.GameSpec.newBuilder()
    TextFormat.getParser().merge(str, builder)
    val game = builder.build()


    println(game.name)
    println(game.boardSize)

    val gameBoard = Array( game.boardSize,{ Array(game.boardSize,{0}) } )


    game.pieceList.forEachIndexed { index, piece ->
        piece.placementList.forEach { placement ->
            val coord = placement.substring(1).split("y").map { it.toInt() }

            if (gameBoard[coord[1]-1][coord[0]-1]==0) {

                gameBoard[coord[1]-1][coord[0]-1] = index+1
                gameBoard[game.boardSize-coord[1]][game.boardSize-coord[0]] = -index-1

            } else throw InvalidStateException("there is already a piece at $coord")
        }
    }

    println("\nboard:\n")

    printArray(gameBoard)
    getBoardStates(gameBoard)

}


