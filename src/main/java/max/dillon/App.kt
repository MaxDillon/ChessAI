package max.dillon

import com.google.protobuf.TextFormat
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Paths



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
            if (gameBoard[coord[0]][coord[1]]==0) {

            }
        }
    }

}

