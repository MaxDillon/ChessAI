package maximum.industries

import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import java.lang.Math.pow
import java.nio.FloatBuffer
import java.security.SecureRandom
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

var rand = Random(SecureRandom().nextLong())

// Extension methods for type-safe equality checking for similar enums, so compiler
// will prevent mistakes like (inst.player == state.player)
fun Player.eq(other: Player): Boolean = this == other

fun Instance.Player.eq(other: Instance.Player): Boolean = this == other

class MonteCarloTreeSearch(val strategy: MctsStrategy,
                           val params: SearchParameters) : GameSearchAlgo {
    override fun next(state: GameState): Pair<GameState, SlimState?> {
        assert(state.outcome == Outcome.UNDETERMINED)
        val stack = ArrayList<GameState>()

        for (i in 0..params.iterations) {
            var node = state
            while (strategy.expanded(node) && node.outcome == Outcome.UNDETERMINED) {
                stack.add(node)
                node = node.nextMoves.maxBy { next ->
                    strategy.searchPriority(node, next)
                }!!
            }
            strategy.expand(node) // may be called multiple times for terminal nodes
            strategy.backprop(stack, node)
            stack.clear()
        }
        println("Cache size: ${strategy.cacheSize()}")
        return strategy.pickMove(state)
    }

    override fun gameOver() {
        strategy.gameOver()
    }
}

data class SlimState(val state: ByteArray,
                     val player: Instance.Player,
                     val treeSearchResults: Array<Instance.TreeSearchResult>)

interface MctsStrategy {
    // An expanded node is one that has at least an initial value estimate (via a
    // rollout or model eval) and whose children, if any, have priors (and initialized
    // values for terminal nodes. The algorithm will expand one node per iteration.
    fun expanded(state: GameState): Boolean

    // Returns a priority balancing three components:
    // (1) the current estimated value of the node (incl. prior & model/mcts est.)
    // (2) the information value of exploring the node (zero for terminal nodes)
    // (3) the bias value of an immediate terminal win or loss.
    fun searchPriority(s1: GameState, s2: GameState): Double

    // For a terminal node just increments visit count and sets fixed value. For a
    // non-terminal node initializes child priors and sets initial node value either
    // via rollout or model. During rollout immediate winning moves must be taken
    // and immediate losing moves avoided if possible.
    fun expand(state: GameState)

    // Backprop estimated values of just-expanded nodes to the root of the search.
    fun backprop(stack: List<GameState>, expanded: GameState)

    // Picks a move according to accumulated mcts stats & prunes cache
    fun pickMove(state: GameState): Pair<GameState, SlimState>

    // Clear any caches
    fun gameOver()

    // Report on size of caches
    fun cacheSize(): Int
}

open class VanillaMctsStrategy(val params: SearchParameters) : MctsStrategy {
    open class Info(var expanded: Boolean = false,
                    var N: Int = 0,
                    var Q: Float = 0f,
                    var P: Float = 0f)

    val allInfo = HashMap<Short, HashMap<GameState, Info>>()

    open fun info(state: GameState): Info {
        val depthInfo = allInfo.getOrPut(state.moveDepth) { HashMap() }
        return depthInfo.getOrPut(state) { Info() }
    }

    override fun cacheSize(): Int {
        var size = 0
        for ((_, map) in allInfo) {
            size += map.size
        }
        return size
    }

    fun sign(s1: GameState, s2: GameState): Int {
        // for performance not using .eq since clearly no type confusion here
        return if (s1.player == s2.player) 1 else -1
    }

    fun GameState.initialSelfValue(): Float {
        return when (outcome) {
            Outcome.WIN -> 1f
            Outcome.LOSE -> -1f
            Outcome.DRAW -> 0f
            else -> rand.nextFloat() * 0.0001f
        }
    }

    override fun expanded(state: GameState) = info(state).expanded

    override fun searchPriority(s1: GameState, s2: GameState): Double {
        val s1Info = info(s1)
        val s2Info = info(s2)
        // the prior is the model's estimate of the probability we'll make this move.
        // we'll treat it here as a component of expected node value. TODO: tune
        // note: this is not currently normalized. may have sum>1 or sum<1.
        val priorValue = s2Info.P * 2.0 / sqrt(4.0 + s2Info.N)
        // the estimated child node value, sign adjusted to be value for current player
        val nodeValue = sign(s1, s2) * s2Info.Q
        val termValue =
                if (s2.winFor(s1)) 100.0 // during search always take immediate wins
                else if (s2.lossFor(s1)) -100.0 else 0.0 // and avoid immediate losses
        val infoValue =
                if (s2.outcome != Outcome.UNDETERMINED) 0.0 // no info value for terminals
                else params.exploration * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
        return priorValue + nodeValue + termValue + infoValue
    }

    // picks a child node, first taking a win if one exists, then taking non-loss
    // moves according to the given weights, then taking a loss if necessary.
    fun pickChildByProbs(state: GameState, weights: DoubleArray): GameState {
        val wins = state.nextMoves.filter { it.winFor(state) }
        if (wins.isNotEmpty()) {
            // if there's a terminal win choose one
            return wins[rand.nextInt(wins.size)]
        } else {
            val sz = state.nextMoves.size
            val nonLossCumls = DoubleArray(sz) {
                if (state.nextMoves[it].lossFor(state)) 0.0 else weights[it]
            }
            for (i in 1 until sz) nonLossCumls[i] += nonLossCumls[i - 1]
            if (nonLossCumls.last() == 0.0) {
                // there are no non-loss moves. return a random loss.
                return state.nextMoves[rand.nextInt(sz)]
            }
            val point = rand.nextFloat() * nonLossCumls.last()
            for (i in 0 until sz) if (nonLossCumls[i] >= point) return state.nextMoves[i]
            // in principle the next line will never be reached
            return state.nextMoves.last()
        }
    }

    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.P = 1f
            }
            var rollout = state
            while (rollout.outcome == Outcome.UNDETERMINED) {
                // use uniform probability for non-loss moves
                rollout = pickChildByProbs(rollout, DoubleArray(rollout.nextMoves.size) { 1.0 })
            }
            sInfo.Q = sign(state, rollout) * rollout.initialSelfValue()
        }
        sInfo.N += 1
        sInfo.expanded = true
    }

    override fun backprop(stack: List<GameState>, expanded: GameState) {
        val eInfo = info(expanded)
        for (ancestor in stack) {
            val aInfo = info(ancestor)
            aInfo.Q = (aInfo.Q * aInfo.N + eInfo.Q * sign(ancestor, expanded)) / (aInfo.N + 1)
            aInfo.N += 1
        }
    }

    fun temperature(moveDepth: Short): Double {
        if (params.rampBy <= 1) {
            return params.temperature
        } else {
            val initial = 1.0
            val final = params.temperature
            val move = moveDepth / 2 + 1
            val mix = min(1.0, move.toDouble() / params.rampBy)
            return (1.0 - mix) * initial + mix * final
        }
    }

    // maybe some auto-pruning of caches at level N+4 or so?
    // problem with P * exploration is we'll never explore things with low prior.
    // Q: should the move picking use Q and/or P in addition to N? lots of times we have ties
    // for N because the branching factor is high and exploration is sufficiently high to
    // make us spread across everything. maybe the fix we need is rather in the priority.
    // Q: should we have a more auto-tuned concept of temperature where we enforce a maximum
    // entropy?
    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        println("Value: ${info(state).Q}")
        val sz = state.nextMoves.size
        val policy = DoubleArray(sz) { info(state.nextMoves[it]).N.toDouble() }
        var policySum = policy.sum()
        for (i in 0 until sz) policy[i] /= policySum
        // record non-exponentiated normalized policy in slim state. we don't want
        // model inputs to depend on temperature, or have most values driven to zero.
        val slim = state.toSlimState { i, tsr -> tsr.prob = policy[i].toFloat() }
        // now exponentiate to get weights for picking actual move
        val inverseTemp = 1 / temperature(state.moveDepth)
        for (i in 0 until sz) policy[i] = pow(policy[i], inverseTemp)
        policySum = policy.sum()
        for (i in state.nextMoves.indices) {
            val next = state.nextMoves[i]
            val nInfo = info(next)
            println("$next:\t${(policy[i]/policySum).toFloat().f3()}\t${nInfo.N}\t${nInfo.Q.f3()}\t${nInfo.P.f3()}")
        }
        val next = pickChildByProbs(state, policy)
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        allInfo.remove((state.moveDepth - 1).toShort()) // or these, which might exist if there are two algos
        return Pair(next, slim)
    }

    override fun gameOver() {
        allInfo.clear()
    }
}

open class AlphaZeroMctsNoModelStrategy0(params: SearchParameters) : VanillaMctsStrategy(params) {
    override fun searchPriority(s1: GameState, s2: GameState): Double {
        val s1Info = info(s1)
        val s2Info = info(s2)
        // the estimated child node value, sign adjusted to be value for current player
        val nodeValue = sign(s1, s2) * s2Info.Q
        val termValue =
                if (s2.winFor(s1)) 100.0 // during search always take immediate wins
                else if (s2.lossFor(s1)) -100.0 else 0.0 // and avoid immediate losses
        val infoValue =
                if (s2.outcome != Outcome.UNDETERMINED) 0.0 // no info value for terminals
                else params.exploration * s2Info.P * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
        return nodeValue + termValue + infoValue
    }
}

open class AlphaZeroMctsStrategy0(val model: ComputationGraph, params: SearchParameters) :
        AlphaZeroMctsNoModelStrategy0(params) {
    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toModelInput())
            val output_value = outputs[0]
            val output_policy = outputs[1]
            sInfo.Q = output_value.getFloat(0 /* batch index */)
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.P = output_policy.getFloat(intArrayOf(0 /* batch index */,
                                                            next.toPolicyIndex()))
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }
}

open class AlphaZeroMctsNoModelStrategy(params: SearchParameters) : VanillaMctsStrategy(params) {
    override fun searchPriority(s1: GameState, s2: GameState): Double {
        val s1Info = info(s1)
        val s2Info = info(s2)
        // the estimated child node value, sign adjusted to be value for current player
        val nodeValue = sign(s1, s2) * s2Info.Q
        val termValue =
                if (s2.winFor(s1)) 100.0 // during search always take immediate wins
                else if (s2.lossFor(s1)) -100.0 else 0.0 // and avoid immediate losses
        val infoValue =
                if (s2.outcome != Outcome.UNDETERMINED) 0.0 // no info value for terminals
                else params.exploration / 2 *
                     pow(sqrt(s1Info.N.toDouble()) / (1.0 + s2Info.N), params.priority_exponent) *
                     (params.priority_uniform / s1.nextMoves.size + s2Info.P)
        return nodeValue + termValue + infoValue
    }
}

open class AlphaZeroMctsStrategy(val model: ComputationGraph, params: SearchParameters) :
        AlphaZeroMctsNoModelStrategy(params) {

    fun evalState(state: GameState): Pair<INDArray, INDArray> {
        val outputs = model.output(state.toModelInput())
        val output_value = outputs[0]
        val output_policy = outputs[1]
        return Pair(output_value, output_policy)
    }

    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val (output_value, output_policy) = evalState(state)
            sInfo.Q = output_value.getFloat(0 /* batch index */)
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.P = output_policy.getFloat(intArrayOf(0 /* batch index */,
                                                            next.toPolicyIndex()))
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }
}

open class AlphaZeroTensorFlowMctsStrategy(val model: Graph, params: SearchParameters) :
        AlphaZeroMctsNoModelStrategy(params) {
    val session = Session(model)
    val reflections = intArrayOf(0,1,2,3)

    public fun evalState(state: GameState): Pair<FloatBuffer, FloatBuffer> {
        val input = state.toTensorInput(reflections)
        val results =
                session.runner()
                        .feed("input", input)
                        .fetch("value/Tanh")
                        .fetch("policy/Softmax").run()
        val output_value = results.get(0)
        val output_policy = results.get(1)
        val value_buf = FloatBuffer.allocate(output_value.numElements())
        val policy_buf = FloatBuffer.allocate(output_policy.numElements())
        output_value.writeTo(value_buf)
        output_policy.writeTo(policy_buf)

        input.close()
        output_value.close()
        output_policy.close()
        return Pair(value_buf, policy_buf)
    }

    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val (value_buf, policy_buf) = evalState(state)
            var Qsum = 0.0f
            for (i in 0 until reflections.size) Qsum += value_buf[i]
            sInfo.Q = Qsum / reflections.size
            val policySize = state.gameSpec.policySize()
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.P = policy_buf.get(next.toPolicyIndex())
                var PSum = 0f
                for (i in 0 until reflections.size) {
                    val flipLeftRight = reflections[i] % 2 > 0
                    val reverseSides = reflections[i] / 2 > 0
                    val policyIndex = state.gameSpec.flipPolicyIndex(
                            next.toPolicyIndex(), flipLeftRight, reverseSides)
                    PSum += policy_buf.get(i * policySize + policyIndex)
                }
                nInfo.P = PSum / reflections.size
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }
}

open class AlphaZeroMctsStrategy1(model: ComputationGraph, params: SearchParameters) :
        AlphaZeroMctsStrategy(model, params) {
    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toModelInput())
            val output_value = outputs[0]
            val output_policy = outputs[1]
            sInfo.Q = output_value.getFloat(0 /* batch index */)
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                // initialize Q's based on parent estimate
                if (next.outcome == Outcome.UNDETERMINED) {
                    nInfo.Q += sign(state, next) * sInfo.Q * 0.9f
                }
                nInfo.P = output_policy.getFloat(intArrayOf(0 /* batch index */,
                                                            next.toPolicyIndex()))
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }

    override fun backprop(stack: List<GameState>, expanded: GameState) {
        val eInfo = info(expanded)
        for (ancestor in stack) {
            val aInfo = info(ancestor)
            // discount distant payoffs to count as closer to mean of 0.0
            val diff = expanded.moveDepth - ancestor.moveDepth
            val discount = Math.pow(0.95, diff.toDouble()).toFloat()
            aInfo.Q = (aInfo.Q * aInfo.N + discount * eInfo.Q * sign(ancestor, expanded)) / (aInfo.N + 1)
            aInfo.N += 1
        }
    }

    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        println("Value: ${info(state).Q}")
        val sz = state.nextMoves.size
        val policy = DoubleArray(sz) {
            // break ties in counts using P
            max(0.0,
                info(state.nextMoves[it]).N.toDouble() + info(state.nextMoves[it]).P)
        }
        var policySum = policy.sum()
        for (i in 0 until sz) policy[i] /= policySum
        // record non-exponentiated normalized policy in slim state. we don't want
        // model inputs to depend on temperature, or have most values driven to zero.
        val slim = state.toSlimState { i, tsr -> tsr.prob = policy[i].toFloat() }
        // now exponentiate to get weights for picking actual move
        val inverseTemp = 1 / temperature(state.moveDepth)
        for (i in 0 until sz) policy[i] = pow(policy[i], inverseTemp)
        policySum = policy.sum()
        for (i in state.nextMoves.indices) {
            val next = state.nextMoves[i]
            val nInfo = info(next)
            println("$next:\t${(policy[i]/policySum).toFloat().f3()}\t${nInfo.N}\t${nInfo.Q.f3()}\t${nInfo.P.f3()}")
        }
        val next = pickChildByProbs(state, policy)
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        allInfo.remove((state.moveDepth - 1).toShort()) // or these, which might exist if there are two algos
        return Pair(next, slim)
    }
}

open class AlphaZeroMctsStrategy2(model: ComputationGraph, params: SearchParameters):
    AlphaZeroMctsStrategy1(model, params) {
    val reflections = intArrayOf(0,1,2,3)

    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toModelInput(reflections))
            val output_values = outputs[0]
            val output_policies = outputs[1]
            sInfo.Q = output_values.meanNumber().toFloat()
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                // initialize Q's based on parent estimate
                // TODO: should this decay like we have above?
                if (next.outcome == Outcome.UNDETERMINED) {
                    nInfo.Q += sign(state, next) * sInfo.Q
                }
                var PSum = 0f
                for (i in 0 until reflections.size) {
                    val flipLeftRight = reflections[i] % 2 > 0
                    val reverseSides = reflections[i] / 2 > 0
                    PSum += output_policies.getFloat(
                            intArrayOf(i, state.gameSpec.flipPolicyIndex(
                                    next.toPolicyIndex(), flipLeftRight, reverseSides)))
                }
                nInfo.P = PSum / reflections.size
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }
}

open class AlphaZeroMctsStrategy3(model: ComputationGraph, params: SearchParameters):
    AlphaZeroMctsStrategy2(model, params) {
    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        println("Value: ${info(state).Q}")
        val sz = state.nextMoves.size
        val policy = DoubleArray(sz) {
            // break ties in counts using P
            max(0.0,
                info(state.nextMoves[it]).N.toDouble() + info(state.nextMoves[it]).P)
        }
        var policySum = policy.sum()
        for (i in 0 until sz) policy[i] /= policySum
        // record non-exponentiated normalized policy in slim state. we don't want
        // model inputs to depend on temperature, or have most values driven to zero.
        val slim = state.toSlimState { i, tsr -> tsr.prob = policy[i].toFloat() }
        val temp = temperature(state.moveDepth)

        for (i in 0 until sz) {
            policy[i] = (policy[i] * 0.1
                         - info(state.nextMoves[i]).Q * 5 * (0.2 + Math.abs(info(state).Q))
                         + rand.nextFloat() * 0.1 * temp * temp)
        }
        var maxi = 0
        for (i in state.nextMoves.indices) {
            if (policy[i] > policy[maxi]) maxi = i
            val next = state.nextMoves[i]
            val nInfo = info(next)
            println("$next:\t${(policy[i]).toFloat().f3()}\t${nInfo.N}\t${nInfo.Q.f3()}\t${nInfo.P.f3()}")
        }
        val next = state.nextMoves[maxi]
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        allInfo.remove((state.moveDepth - 1).toShort()) // or these, which might exist if there are two algos
        return Pair(next, slim)
    }
}

class DirichletMctsStrategy(params: SearchParameters, val values: FloatArray) :
        VanillaMctsStrategy(params) {
    open class DirichletInfo : Info() {
        var wld = FloatArray(3) { 0f }
    }

    override fun info(state: GameState): DirichletInfo {
        val depthInfo = allInfo.getOrPut(state.moveDepth) { HashMap() }
        return depthInfo.getOrPut(state) { DirichletInfo() } as DirichletInfo
    }

    fun GameState.initialSelfDirichlet(): FloatArray {
        return when (outcome) {
            Outcome.WIN -> floatArrayOf(1f, 0f, 0f)
            Outcome.LOSE -> floatArrayOf(0f, 1f, 0f)
            Outcome.DRAW -> floatArrayOf(0f, 0f, 1f)
            else -> floatArrayOf(0.1f, 0.1f, 0.1f)
        }
    }

    fun expectedValuePlusSdevs(wld: FloatArray, sdevs: Float): Float {
        val n = wld.sum()
        val e = values[0] * wld[0] / n +
                values[1] * wld[1] / n +
                values[2] * wld[2] / n
        var varev = 0f  // accumulator for variance of expected value
        val vdenom = n * n * (n + 1)  // dirichlet var/cov denominator
        for (i in 0..2) for (j in 0..2) {
            val cov_ij =
                    if (i == j) wld[i] * (n - wld[i]) / vdenom
                    else -wld[i] * wld[j] / vdenom
            varev += cov_ij * values[i] * values[j]
        }
        return e + sqrt(varev) * sdevs
    }

    override fun searchPriority(s1: GameState, s2: GameState): Double {
        val s2Info = info(s2)
        val nodeValue = sign(s1, s2) * expectedValuePlusSdevs(s2Info.wld, 4f)
        val winValue = if (s2.winFor(s1)) 100.0 else 0.0 // always take immed. wins
        return nodeValue + winValue
    }

    override fun expand(state: GameState) {
        state.protectNextMoves()
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            for (next in state.nextMoves) {
                info(next).wld = next.initialSelfDirichlet()
            }
            var rollout = state
            while (rollout.outcome == Outcome.UNDETERMINED) {
                // use uniform probability for non-loss moves
                rollout = pickChildByProbs(rollout, DoubleArray(rollout.nextMoves.size) { 1.0 })
            }
            val result = rollout.initialSelfDirichlet()
            // TODO: UMMM .... this seems wrong:
            for (i in 0 until 3) sInfo.wld[i] += sign(state, rollout) * result[i]
        }
        sInfo.N += 1
        sInfo.expanded = true
    }

    override fun backprop(stack: List<GameState>, expanded: GameState) {
        val eInfo = info(expanded)
        for (ancestor in stack) {
            val aInfo = info(ancestor)
            if (sign(ancestor, expanded) > 0) {
                aInfo.wld[0] += eInfo.wld[0]
                aInfo.wld[1] += eInfo.wld[1]
            } else {
                aInfo.wld[0] += eInfo.wld[1]
                aInfo.wld[1] += eInfo.wld[0]
            }
            aInfo.wld[2] += eInfo.wld[2]
        }
    }

    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        for (next in state.nextMoves) {
            val nInfo = info(next)
        }
        val sz = state.nextMoves.size
        val values = DoubleArray(sz) {
            pow(expectedValuePlusSdevs(info(state.nextMoves[it]).wld, -1f).toDouble(),
                1 / temperature(state.moveDepth))
        }
        val next = pickChildByProbs(state, values)
        val slim = state.toSlimState { i, tsr ->
            val wld = info(state.nextMoves[i]).wld
            val wldsum = max(wld.sum(), 0.001f)
            tsr.wld = Instance.WinLoseDraw.newBuilder()
                    .setWin(wld[0] / wldsum)
                    .setLose(wld[1] / wldsum) // TODO: do we need to swap win/loss here?
                    .setDraw(wld[2] / wldsum)
                    .build()
        }
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        return Pair(next, slim)
    }
}
