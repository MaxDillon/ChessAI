package max.dillon

import max.dillon.Instance.TrainingInstance
import java.io.FileInputStream

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    var count = 0
    var error = 0
    val instream = FileInputStream(args[1])
    while (true) {
        try {
            val inst = TrainingInstance.parseDelimitedFrom(instream)
            if (inst == null) {
                break
            }
            count += 1

            var state = GameState(gameSpec)
            val sz = gameSpec.boardSize
            state.whiteMove = inst.whiteMove
            for (i in 0 until inst.boardState.size()) {
                val x = i / sz
                val y = i % sz
                state.gameBoard[y][x] = inst.boardState.byteAt(i).toInt()
            }

            state.printBoard()
            for (tsr in inst.treeSearchResultList) {
                println("${tsr.index}: ${tsr.prob.f3()}")
            }
            println("STATE")
        } catch (e: Exception) {
            error += 1
        }
    }
    println("Read ${count} states. ${error} errors.")
}
