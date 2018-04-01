package max.dillon

import com.google.protobuf.TextFormat
import max.dillon.GameGrammar.GameSpec
import max.dillon.Instance.TrainingInstance
import java.io.FileInputStream
import java.nio.file.Files
import java.nio.file.Paths

fun main(args: Array<String>) {
    val specStr = String(Files.readAllBytes(Paths.get("src/main/data/${args[0]}.textproto")))
    val builder = GameSpec.newBuilder()
    TextFormat.getParser().merge(specStr, builder)
    val gameSpec = builder.apply {
        addPiece(0, builder.addPieceBuilder())
    }.build()

    val instream = FileInputStream(args[1])
    while (true) {
        val inst = TrainingInstance.parseDelimitedFrom(instream)
        if (inst == null) {
            break
        }

        var state = GameState(gameSpec)
        val sz = gameSpec.boardSize
        state.whiteMove = inst.whiteMove
        for (i in 0 until inst.boardState.size()) {
            val x = i / sz
            val y = i % sz
            state.gameBoard[y][x] = inst.boardState.byteAt(i).toInt()
        }

        state.printBoard()
    }
}
