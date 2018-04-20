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
            val gameBoard = Array(sz) { IntArray(sz) { 0 } }
            for (i in 0 until inst.boardState.size()) {
                val x = i / sz
                val y = i % sz
                gameBoard[y][x] = inst.boardState.byteAt(i).toInt()
            }

            val state = GameState(gameSpec, gameBoard, player,
                                  0,0, 0, 0, 0, 0)
            state.printBoard()
            println("Player: ${inst.player}")
            println("Outcome: ${inst.outcome}")
            println("Length: ${inst.gameLength}")
            for (tsr in inst.treeSearchResultList) {
                if (tsr.type == Instance.TsrType.MOVE_PROB) {
                    println("${tsr.index}:\t${tsr.prob.f3()} ")
                } else {
                    println("${tsr.index}:\t${tsr.wld.win.f3()} ${tsr.wld.lose.f3()} ${tsr.wld.draw.f3()} ")
                }
            }
        } catch (e: Exception) {
            error += 1
        }
    }
    println("Read $states states. $error errors.")
}
