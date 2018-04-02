package max.dillon

import com.google.protobuf.TextFormat
import max.dillon.GameGrammar.GameSpec
import max.dillon.Instance.TrainingInstance
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Paths

fun main(args: Array<String>) {
    val specStr = String(Files.readAllBytes(Paths.get("src/main/data/${args[0]}.textproto")))
    val builder = GameSpec.newBuilder()
    TextFormat.getParser().merge(specStr, builder)
    val gameSpec = builder.apply {
        addPiece(0, builder.addPieceBuilder())
    }.build()

    // needs to applied to cases where we have computed the source index as <piece> + <position>
    // where position is in NxN and we've done this for games like chess where we never move a
    // piece from off the board.  In these cases we should subtract numPieces from the src index.
    //
    // also for cases where we move from offboard we should exclude the virtual zero piece.
    fun fixIndex(idx: Int): Int {
        val sz = gameSpec.boardSize
        val np = gameSpec.pieceCount
        val src = idx / (sz * sz)
        val dst = idx % (sz * sz)
        if (gameSpec.moveSource == GameGrammar.MoveSource.PIECES_ON_BOARD) {
            assert(src >= np)
            return (src - np) * (sz * sz) + dst
        } else {
            assert(src > 0)
            assert(src < np)
            assert(gameSpec.pieceList[0].name == "")
            return (src - 1) * (sz * sz) + dst
        }
    }

    val instream = FileInputStream(args[1])
    val outstream = FileOutputStream(args[2])
    while (true) {
        val inst = TrainingInstance.parseDelimitedFrom(instream)
        if (inst == null) {
            break
        }
        TrainingInstance.newBuilder().apply {
            boardState = inst.boardState
            whiteMove = inst.whiteMove
            gameLength = inst.gameLength
            outcome = inst.outcome
            for (tsr in inst.treeSearchResultList) {
                addTreeSearchResultBuilder().apply {
                    index = fixIndex(tsr.index)
                    meanValue = tsr.meanValue
                    numVisits = tsr.numVisits
                    prior = tsr.prior
                }
            }
        }.build().writeDelimitedTo(outstream)
    }
}
