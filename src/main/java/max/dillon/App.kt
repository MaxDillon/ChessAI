package max.dillon


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


fun plus(x: Int, y: Int, length: Int) {

}

fun cross(x: Int, y: Int, length: Int) {

}

fun square(x: Int, y: Int, length: Int) {

}


fun getPieceMoves(game: GameGrammar.GameSpec, x: Int, y: Int, num: Int): Array<Int> {
    val newBoard = Array( game.boardSize,{ Array(game.boardSize,{0}) } )
    val piece = game.getPiece(num)
    piece.moveList.forEach {
        it.templateList.forEach {
            val (a,b,c) = it.substring(1).split("_")


        }
    }
    TODO("finish")

}


fun getBoardStates(game:GameGrammar.GameSpec, board: Array<Array<Int>>) {
    val possibleStates = arrayListOf<Int>()
    board.forEachIndexed { x, list->
        list.forEachIndexed { y, piece ->
            if(piece != 0) possibleStates.addAll(getPieceMoves(game,x,y,piece))

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
            val coord = placement.substring(1).split("y").map { it.toInt()-1 }
            println(coord)
            if ((gameBoard[coord[1]][coord[0]] == 0 )) {
                gameBoard[coord[1]][coord[0]]=1


            } else throw InvalidStateException("you cant place a piece at $coord")
        }
    }

    println("\nboard:\n")

    printArray(gameBoard)
    getBoardStates(game,gameBoard)

}


