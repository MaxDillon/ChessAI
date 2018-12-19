package maximum.industries

import maximum.industries.GameGrammar.GameSpec
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.conditions.GreaterThan
import org.nd4j.linalg.lossfunctions.impl.PseudoSpherical
import kotlin.math.sqrt
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.shade.jackson.databind.jsontype.NamedType
import java.util.*

fun checkModelConsistency(gameSpec: GameSpec, model: ComputationGraph,
                          maxStates: Int, maxBatch: Int, minDepth: Int, maxDepth: Int) {
    println(" *\tPlayer\t+Mass\tCorrel\t+Pos\t-Pos\tFrac+")
    checkModelConsistency(gameSpec, model, maxStates, maxBatch, minDepth, maxDepth, Player.WHITE)
    checkModelConsistency(gameSpec, model, maxStates, maxBatch, minDepth, maxDepth, Player.BLACK)
}

// Even though we may be making predictions for a much smaller set of examples, we want to
// provide the model with a full batch. This is both faster and maybe necessary to prevent
// GPU memory growth/exhaustion as the Dl4j will cache tensors of each shape on the GPU.
// although ... this doesn't seem to fix the GPU memory problem
fun emBatchen(inputs: Array<INDArray>, batchSize: Long): INDArray {
    var shape = IntArray(4) { if (it == 0) batchSize.toInt() else inputs[0].shape()[it].toInt() }
    val input = Nd4j.zeros(*shape)
    val indices = Array(shape.size) { NDArrayIndex.all() }
    for (i in inputs.indices) {
        if (i < batchSize) {
            indices[0] = NDArrayIndex.point(i.toLong())
	    input.put(indices, inputs[i])
        }
    }
    return input
}

fun checkModelConsistency(gameSpec: GameSpec, model: ComputationGraph,
                          maxStates: Int, maxBatch: Int, minDepth: Int, maxDepth: Int,
                          player: Player) {
    val doValue = modelHas(model, "value")
    val doLegal = modelHas(model, "legal")

    var numStates = 0
    var numNextLegal = 0
    var numNextIllegal = 0
    var numNextLegalPos = 0
    var numNextIllegalPos = 0
    var sumNextLegalMass = 0f
    var sumNextIllegalMass = 0f
    var sumCorrelation = 0f
    var numCorrelated = 0

    while (numStates < maxStates) {
        var state = newGame(gameSpec)
        while (state.outcome == Outcome.UNDETERMINED && state.moveDepth < maxDepth && numStates < maxStates) {
            if (player.eq(state.player) && state.moveDepth >= minDepth && state.nextMoves.size < maxBatch) {
                numStates += 1
                val numNext = state.nextMoves.size
                // get prediction for target player's current move
                val inputs = Array(numNext + 1) {
                    if (it == 0) state.toModelInput()
                    else state.nextMoves[it - 1].toModelInput()
                }
                val outputs = model.output(emBatchen(inputs, maxBatch.toLong()))

                val policy = outputs[if (doValue) 1 else 0].getRow(0)
                val isLegal = Nd4j.zeros(policySize(gameSpec))
                for (next in state.nextMoves) isLegal.putScalar(next.toPolicyIndex().toLong(), 1f)
                val notLegal = isLegal.sub(1f).muli(-1)
                numNextLegal += numNext
                numNextIllegal += notLegal.sumNumber().toInt()
                sumNextLegalMass += policy.mul(isLegal).sumNumber().toFloat()
                sumNextIllegalMass += policy.mul(notLegal).sumNumber().toFloat()
                val legalPolicy = policy.mul(isLegal)
                val nextVal = Nd4j.zeros(policySize(gameSpec))
                if (doValue) {
                    for (i in state.nextMoves.indices) {
                        nextVal.putScalar(state.nextMoves[i].toPolicyIndex().toLong(), outputs[0].getFloat(i + 1L, 0))
                    }
                }
                val legalNextVal = nextVal.mul(isLegal)

                if (numNext > 2) {
                    val sumProb = legalPolicy.sumNumber().toFloat()
                    val sumValue = legalNextVal.sumNumber().toFloat()
                    val sumProbSq = legalPolicy.mul(legalPolicy).sumNumber().toFloat()
                    val sumValueSq = legalNextVal.mul(legalNextVal).sumNumber().toFloat()
                    val sumProbVal = legalPolicy.mul(legalNextVal).sumNumber().toFloat()
                    val covariance = sumProbVal / numNext - (sumProb / numNext) * (sumValue / numNext)
                    val sdevProb = sqrt(sumProbSq / numNext - (sumProb / numNext) * (sumProb / numNext))
                    val sdevValue = sqrt(sumValueSq / numNext - (sumValue / numNext) * (sumValue / numNext))
                    if (sumProb != 0f && sumValue != 0f && sdevProb > 0 && sdevValue > 0) {
                        sumCorrelation += covariance / sdevProb / sdevValue
                        numCorrelated += 1
                    }
                }

                if (doLegal) {
                    val estLegal = outputs[2].getRow(0)
                    numNextLegalPos += estLegal.mul(isLegal).cond(GreaterThan(0f)).sumNumber().toInt()
                    numNextIllegalPos += estLegal.mul(notLegal).cond(GreaterThan(0f)).sumNumber().toInt()
                }
            }
            if (state.nextMoves.size == 0) state.printBoard()
            state = state.nextMoves[rand.nextInt(state.nextMoves.size)]
        }
    }

    val legalPosFrac = numNextLegalPos / numNextLegal.toFloat()
    val illegalPosFrac = numNextIllegalPos / numNextIllegal.toFloat()
    val avgLegalMass = sumNextLegalMass / numStates
    val avgIllegalMass = sumNextIllegalMass / numStates
    val legalFrac = numNextLegal.toFloat() / (numNextLegal + numNextIllegal)
    var correlation = sumCorrelation / numCorrelated

    print(" *\t$player\t${avgLegalMass.f3()}\t${correlation.f3()}\t")
    print("${legalPosFrac.f3()}\t${illegalPosFrac.f3()}\t${legalFrac.f3()}")
    println()
}

fun main(args: Array<String>) {
//    NeuralNetConfiguration.reinitMapperWithSubtypes(
//            Collections.singletonList(NamedType(PseudoSpherical::class.java)))

    val gameSpec = loadSpec(args[0])
    val model = ModelSerializer.restoreComputationGraph(args[1])

    val mindepth = getArg(args, "mindepth")?.toInt() ?: 20
    val maxdepth = getArg(args, "maxdepth")?.toInt() ?: 75
    val maxbatch = getArg(args, "maxbatch")?.toInt() ?: 75
    val numstates = getArg(args, "states")?.toInt() ?: 500

    checkModelConsistency(gameSpec, model, numstates, maxbatch, mindepth, maxdepth)
    println("##############################################################")
}