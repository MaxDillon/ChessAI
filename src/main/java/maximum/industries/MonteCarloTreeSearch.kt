package maximum.industries

import maximum.industries.GameGrammar.MoveSource
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.lang.Math.pow
import java.security.SecureRandom
import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

val rand = Random(SecureRandom().nextLong())

// Extension methods for type-safe equality checking for similar enums, so compiler
// will prevent mistakes like (inst.player == state.player)
fun Player.eq(other: Player): Boolean = this == other
fun Instance.Player.eq(other: Instance.Player): Boolean = this == other

class MonteCarloTreeSearch(val strategy: MctsStrategy,
                           val iterations: Int) : GameSearchAlgo {
    override fun next(state: GameState): Pair<GameState, SlimState?> {
        assert(state.outcome == Outcome.UNDETERMINED)
        val stack = ArrayList<GameState>()

        for (i in 0..iterations) {
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

open class VanillaMctsStrategy(val exploration: Double, val temperature: Double) : MctsStrategy {
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
        return if (s1.player.eq(s2.player)) 1 else -1
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
                else exploration * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
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
            var count = 0;
            for (i in 0 until sz) if (nonLossCumls[i] < point) count++
            return state.nextMoves[count]
        }
    }

    override fun expand(state: GameState) {
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

    // maybe some auto-pruning of caches at level N+4 or so?

    // problem with P * exploration is we'll never explore things with low prior.

    // Q: should the move picking use Q and/or P in addition to N? lots of times we have ties
    // for N because the branching factor is high and exploration is sufficiently high to
    // make us spread across everything. maybe the fix we need is rather in the priority.
    // Q: should we have a more auto-tuned concept of temperature where we enforce a maximum
    // entropy?
    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        println("Value: ${info(state).Q}")
        for (next in state.nextMoves) {
            val nInfo = info(next)
            println("$next:\t${nInfo.N}\t${nInfo.Q.f3()}\t${nInfo.P.f3()}")
        }
        val sz = state.nextMoves.size
        val policy = DoubleArray(sz) { info(state.nextMoves[it]).N.toDouble() }
        val policySum = policy.sum()
        for (i in 0 until sz) policy[i] /= policySum
        // record non-exponentiated normalized policy in slim state. we don't want
        // model inputs to depend on temperature, or have most values driven to zero.
        val slim = state.toSlimState { i, tsr -> tsr.prob = policy[i].toFloat() }
        // now exponentiate to get weights for picking actual move
        for (i in 0 until sz) policy[i] = pow(policy[i], 1 / temperature)
        val next = pickChildByProbs(state, policy)
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        allInfo.remove((state.moveDepth - 1).toShort()) // or these, which might exist if there are two algos
        return Pair(next, slim)
    }

    override fun gameOver() {
        allInfo.clear()
    }
}

open class AlphaZeroMctsNoModelStrategy(exploration: Double, temperature: Double) :
        VanillaMctsStrategy(exploration, temperature) {
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
                else exploration * s2Info.P * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
        return nodeValue + termValue + infoValue
    }
}

open class AlphaZeroMctsStrategy(val model: ComputationGraph,
                                 exploration: Double,
                                 temperature: Double) :
        AlphaZeroMctsNoModelStrategy(exploration, temperature) {
    override fun expand(state: GameState) {
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

class DirichletMctsStrategy(exploration: Double,
                            temperature: Double,
                            val values: FloatArray) :
        VanillaMctsStrategy(exploration, temperature) {

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
                    1 / temperature)
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
