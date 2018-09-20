package maximum.industries

import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import java.io.FileInputStream

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    val sz = gameSpec.boardSize
    var states = 0
    var error = 0
    val instream = FileInputStream(args[1])
    val compare =  getArg(args, "compare")?.toBoolean() ?: false

    var model: ComputationGraph? = null
    if (args.size > 2) {
        model = ModelSerializer.restoreComputationGraph(args[2])
    }

    var prevDepth = 0
    var prevValue = 0f
    while (true) {
        try {
            val inst = Instance.TrainingInstance.parseDelimitedFrom(instream) ?: break
            states += 1

            val (input, out_value, out_policy, out_legal) = initTrainingData(gameSpec, 4)
            for (i in 0..3) {
                inst.toBatchTrainingInput(gameSpec, i.toLong(), i,
                                          input, out_value, out_policy, out_legal)
            }
            val tmpout = model?.output(input)

            val player = if (inst.player.eq(Instance.Player.WHITE)) Player.WHITE else Player.BLACK
            val gameBoard = ByteArray(sz * sz) { 0 }
            for (i in 0 until inst.boardState.size()) {
                val y = i / sz
                val x = i % sz
                gameBoard[y * sz + x] = inst.boardState.byteAt(i)
            }
            val state = GameState(gameSpec, gameBoard, player,
                                  0,0, 0, 0, 0, 0)
            val outputs = model?.output(state.toModelInput())
            val value = if (outputs != null) outputs[0].getFloat(0) else 0f
            val policy = if (outputs != null) outputs[1] else null

            if (prevDepth == inst.gameLength && Math.abs(prevValue + value) > 0.8) {
                println("WTF")
            }
            
            prevDepth = inst.gameLength
            prevValue = value
            
            state.printBoard()
            println("Player: ${inst.player}")
            println("Outcome: ${inst.outcome} ${value.f3()}")
            if (tmpout != null) println(tmpout[0].toString())
            println("Length: ${inst.gameLength}")
            var probSum = 0f
            for (tsr in inst.treeSearchResultList) {
                val info = gameSpec.expandPolicyIndex(tsr.index)
                if (tsr.type == Instance.TsrType.MOVE_PROB) {
                    val estprob = if (policy != null) policy.getFloat(intArrayOf(0, tsr.index)) else 0f
                    println("${info}:\t${tsr.prob.f3()} ${estprob.f3()} ")
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
