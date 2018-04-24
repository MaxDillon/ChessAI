package maximum.industries

import maximum.industries.GameGrammar.GameSpec
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.conditions.GreaterThan
import kotlin.math.sqrt

fun checkModelConsistency(gameSpec: GameSpec, model: ComputationGraph, batchSize: Int, depth: Int) {
    println(" *\tPlayer\t+Mass\t-Mass\tCorrel\t+Pos\t-Pos\tFrac+")
    checkModelConsistency(gameSpec, model, batchSize, depth, Player.WHITE)
    checkModelConsistency(gameSpec, model, batchSize, depth, Player.BLACK)
}

// Even though we may be making predictions for a much smaller set of examples, we want to
// provide the model with a full batch. This is both faster and maybe necessary to prevent
// GPU memory growth/exhaustion as the Dl4j will cache tensors of each shape on the GPU.
// although ... this doesn't seem to fix the GPU memory problem
fun emBatchen(inputs: Array<INDArray>, batchSize: Int): INDArray {
    var shape = inputs[0].shape()
    shape[0] = batchSize
    val input = Nd4j.zeros(*shape)
    val indices = Array(shape.size) { NDArrayIndex.all() }
    for (i in inputs.indices) {
        if (i < batchSize) {
            indices[0] = NDArrayIndex.point(i)
            input.put(indices, inputs[i])
        }
    }
    return input
}

fun checkModelConsistency(gameSpec: GameSpec, model: ComputationGraph, batchSize: Int, depth: Int, player: Player) {
    val doLegal = hasLegal(model)

    val maxStates = 500
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
        var state = GameState(gameSpec)
        while (state.outcome == Outcome.UNDETERMINED && state.moveDepth < 50 && numStates < maxStates) {
            if (player.eq(state.player) && state.moveDepth >= depth) {
                numStates += 1
                val numNext = state.nextMoves.size
                // get prediction for target player's current move
                val inputs = Array(numNext + 1) {
                    if (it == 0) state.toProbModelInput()
                    else state.nextMoves[it - 1].toProbModelInput()
                }
                val outputs = model.output(emBatchen(inputs, batchSize))

                val policy = outputs[1].getRow(0)
                val isLegal = Nd4j.zeros(policySize(gameSpec))
                for (next in state.nextMoves) isLegal.putScalar(next.toPolicyIndex(), 1f)
                val notLegal = isLegal.sub(1f).muli(-1)
                numNextLegal += numNext
                numNextIllegal += notLegal.sumNumber().toInt()
                sumNextLegalMass += policy.mul(isLegal).sumNumber().toFloat()
                sumNextIllegalMass += policy.mul(notLegal).sumNumber().toFloat()
                val legalPolicy = policy.mul(isLegal)
                val nextVal = Nd4j.zeros(policySize(gameSpec))
                for (i in state.nextMoves.indices) {
                    nextVal.putScalar(state.nextMoves[i].toPolicyIndex(), outputs[0].getFloat(i + 1, 0))
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
            state = state.nextMoves[rand.nextInt(state.nextMoves.size)]
        }
    }

    val legalPosFrac = numNextLegalPos / numNextLegal.toFloat()
    val illegalPosFrac = numNextIllegalPos / numNextIllegal.toFloat()
    val avgLegalMass = sumNextLegalMass / numStates
    val avgIllegalMass = sumNextIllegalMass / numStates
    val legalFrac = numNextLegal.toFloat() / (numNextLegal + numNextIllegal)
    var correlation = sumCorrelation / numCorrelated

    print(" *\t$player\t${avgLegalMass.f3()}\t${avgIllegalMass.f3()}\t${correlation.f3()}\t")
    print("${legalPosFrac.f3()}\t${illegalPosFrac.f3()}\t${legalFrac.f3()}")
    println()
}

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    val model = ModelSerializer.restoreComputationGraph(args[1])
    val depth = args[2].toInt()
    // Make sure to use a large enough batch size to encompass the largest set of legal moves we might have
    checkModelConsistency(gameSpec, model, 100, depth)
}