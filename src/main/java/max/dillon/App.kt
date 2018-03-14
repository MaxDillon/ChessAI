package max.dillon


import java.util.Arrays;
import com.google.protobuf.TextFormat
import sun.plugin.dom.exception.InvalidStateException
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.absoluteValue


fun printArray(anArray: Array<Array<Int>>) {
//    println(Arrays.deepToString(anArray))
    anArray.forEach { thing ->
        println(Arrays.deepToString(thing))
    }
}

fun main(args: Array<String>) {
    val str = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))

    val builder = GameGrammar.GameSpec.newBuilder()
    TextFormat.getParser().merge(str, builder)
    val game = builder.build()


    println(game.name)
    println(game.boardSize)
    for (piece in game.pieceList) {
        for (move in piece.moveList) println(piece.name + " " + move.templateList)
    }


    val gameBoard = Array( game.boardSize,{ Array(game.boardSize,{0}) } )


    game.pieceList.forEach { piece ->
        piece.placementList.forEach { placement ->
            val coord = placement.substring(1).split("y").map { it.toInt() }
            println("${piece.name} $coord ")

            if (gameBoard[coord[0]-1][coord[1]-1]==0) {

                gameBoard[coord[0]-1][coord[1]-1] = piece.hashCode().absoluteValue%8
                gameBoard[game.boardSize-coord[0]][game.boardSize-coord[1]] = piece.hashCode().absoluteValue%8

            } else throw InvalidStateException("there is already a piece at $coord")
        }
    }
    printArray(gameBoard)


}

