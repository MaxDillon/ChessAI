package maximum.industries

import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.abs

fun oldTsrIndex(gameSpec: GameGrammar.GameSpec, info: MoveInfo): Int {
   var dim = gameSpec.boardSize
    val dst = info.y2 + dim * info.x2
    val src = if (gameSpec.moveSource == GameGrammar.MoveSource.ENDS) {
        abs(info.p1) - 1 // subtract out the virtual piece added to all gamespecs
    } else {
        info.y1 + info.x1 * dim
    }
    return src * dim * dim + dst
}

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    val instream = FileInputStream(args[1])
    val outstream = FileOutputStream(args[2])

    var states = 0
    var error = 0

    while (true) {
        try {
            val inst = Instance.TrainingInstance.parseDelimitedFrom(instream) ?: break

            val sz = gameSpec.boardSize
            val gameBoard = Array(sz) { IntArray(sz) { 0 } }
            for (i in 0 until inst.boardState.size()) {
                val x = i / sz
                val y = i % sz
                gameBoard[y][x] = inst.boardState.byteAt(i).toInt()
            }
            val player = if (inst.player.eq(Instance.Player.WHITE)) Player.WHITE else Player.BLACK
            val state = GameState(gameSpec, gameBoard, player, 0, 0, 0, 0, 0, 0)
            val slim = state.toSlimState { i, tsr ->
                val info = gameSpec.expandPolicyIndex(tsr.index)
                tsr.prob = inst.treeSearchResultList.filter {
                    oldTsrIndex(gameSpec, info) == it.index
                }.first()!!.prob
            }
            var fixed = slim.toTrainingInstance(inst.outcome, inst.gameLength)
            fixed.writeDelimitedTo(outstream)

            states += 1
        } catch (e: Exception) {
            e.printStackTrace()
            error += 1
        }
    }
    println("Read $states states. $error errors.")
}
