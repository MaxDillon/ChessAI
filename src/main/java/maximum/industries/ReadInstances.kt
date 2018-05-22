package maximum.industries

import java.io.FileInputStream

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    var states = 0
    var error = 0
    val instream = FileInputStream(args[1])

    while (true) {
        try {
            val inst = Instance.TrainingInstance.parseDelimitedFrom(instream) ?: break
            states += 1

            val sz = gameSpec.boardSize
            val player = if (inst.player.eq(Instance.Player.WHITE)) Player.WHITE else Player.BLACK
            val gameBoard = ByteArray(sz * sz) { 0 }
            for (i in 0 until inst.boardState.size()) {
                val y = i / sz
                val x = i % sz
                gameBoard[y * sz + x] = inst.boardState.byteAt(i)
            }
            val state = GameState(gameSpec, gameBoard, player,
                                  0,0, 0, 0, 0, 0)
            state.printBoard()
            println("Player: ${inst.player}")
            println("Outcome: ${inst.outcome}")
            println("Length: ${inst.gameLength}")
            var probSum = 0f
            for (tsr in inst.treeSearchResultList) {
                val info = gameSpec.expandPolicyIndex(tsr.index)
                if (tsr.type == Instance.TsrType.MOVE_PROB) {
                    println("${info}:\t${tsr.prob.f3()} ")
                    probSum += tsr.prob
                } else {
                    println("${info}:\t${tsr.wld.win.f3()} ${tsr.wld.lose.f3()} ${tsr.wld.draw.f3()} ")
                }
            }
            println("ProbSum: ${probSum}")
        } catch (e: Exception) {
            e.printStackTrace()
            error += 1
        }
    }
    println("Read $states states. $error errors.")
}
