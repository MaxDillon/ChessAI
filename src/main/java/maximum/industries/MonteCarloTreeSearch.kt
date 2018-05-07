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

        for (i in 1..iterations) {
            var node = state
            while (strategy.expanded(node) &&
                   node.outcome == Outcome.UNDETERMINED) {
                stack.add(node)
                node = node.nextMoves.maxBy { next ->
                    strategy.searchPriority(node, next)
                }!!
            }
            strategy.expand(node) // may be called multiple times for terminal nodes
            strategy.backprop(stack, node)
            stack.clear()
        }
        return strategy.pickMove(state)
    }

    override fun gameOver() {
        strategy.gameOver()
    }
}

fun GameState.toPolicyIndex(): Int {
    val dim = gameSpec.boardSize
    val dst = y2 + dim * x2
    val src = if (gameSpec.moveSource == MoveSource.ENDS) {
        abs(p1) - 1 // subtract out the virtual piece added to all gamespecs
    } else {
        y1 + x1 * dim
    }
    return src * dim * dim + dst
}

data class SlimState(val state: ByteArray,
                     val player: Instance.Player,
                     val treeSearchResults: Array<Instance.TreeSearchResult>)

interface MctsStrategy {
    // An expanded node is one that has at least an initial value estimate (via a
    // rollout or model eval) and whose children, if any, have priors (and initialized
    // values for terminal nodes0. The algorithm will expand one node per iteration.
    fun expanded(state: GameState): Boolean

    // Returns a priority balancing three components:
    // (1) the current estimated value of the node (incl. prior & model/mcts est.)
    // (2) the information value of exploring the node (zero for terminal nodes)
    // (3) the bias value of an immediate terminal win or loss.
    fun searchPriority(s1: GameState, s2: GameState): Double

    // For a terminal node just increments visit count and sets fixed value. For a
    // non-terminal nodeinitializes child priors and sets initial node value either
    // via rollout or model. During rollout immediate winning moves must be taken.
    fun expand(state: GameState)

    // Backprop estimated values of just-expanded nodes to the root of the search.
    fun backprop(stack: List<GameState>, expanded: GameState)

    // Picks a move according to accumulated mcts stats & prunes cache
    fun pickMove(state: GameState): Pair<GameState, SlimState>

    // Clear any caches
    fun gameOver()
}

open class VanillaMctsStrategy(val exploration: Double, val temperature: Double) : MctsStrategy {
    open class Info(var expanded: Boolean = false,
                    var N: Int = 0,
                    var Q: Float = 0f,
                    var P: Float = 0f)

    val allInfo = HashMap<Int, HashMap<GameState, Info>>()

    open fun info(state: GameState): Info {
        val depthInfo = allInfo.getOrPut(state.moveDepth) { HashMap() }
        return depthInfo.getOrPut(state) { Info() }
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
        val priorValue = s2Info.P * 2.0 / sqrt(4.0 + s2Info.N)
        // the estimated child node value, sign adjusted to be value for current player
        val nodeValue = sign(s1, s2) * s2Info.Q
        val termValue =
                if (s2.winFor(s1)) 100.0 // always take immediate wins
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
            if (nonLossCumls[sz - 1] == 0.0) {
                // there are no non-loss moves. return a random loss.
                return state.nextMoves[rand.nextInt(sz)]
            }
            val point = rand.nextFloat() * nonLossCumls[sz - 1]
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

    fun report(state: GameState) {
        val info = info(state)
        if (info.expanded) {
            for (i in 0 until state.moveDepth) print("  ")
            println("${state}: ${info.N} ${info.Q.f3()}")
            for (next in state.nextMoves) report(next)
        }
    }

    fun slimState(state: GameState,
                  initTsr: (i: Int, builder: Instance.TreeSearchResult.Builder) -> Unit): SlimState {
        val sz = state.gameSpec.boardSize
        val arr = ByteArray(sz * sz) {
            val x = it / sz
            val y = it % sz
            (state.at(x, y)).toByte()
        }
        val tsrs = Array(state.nextMoves.size) {
            Instance.TreeSearchResult.newBuilder().apply {
                index = state.nextMoves[it].toPolicyIndex()
                initTsr(it, this)
            }.build()
        }
        val player = if (state.player.eq(Player.WHITE)) {
            Instance.Player.WHITE
        } else {
            Instance.Player.BLACK
        }
        return SlimState(arr, player, tsrs)
    }

    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        for (next in state.nextMoves) {
            val nInfo = info(next)
        }
        val sz = state.nextMoves.size
        val policy = DoubleArray(sz) { info(state.nextMoves[it]).N.toDouble() }
        val policySum = policy.sum()
        for (i in 0 until sz) policy[i] /= policySum
        // record non-exponentiated normalized policy in slim state. we don't want
        // model inputs to depend on temperature, or have most values driven to zero.
        val slim = slimState(state) { i, tsr -> tsr.prob = policy[i].toFloat() }
        // now exponentiate to get weights for picking actual move
        for (i in 0 until sz) policy[i] = pow(policy[i], 1 / temperature)
        val next = pickChildByProbs(state, policy)
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        return Pair(next, slim)
    }

    override fun gameOver() {
        allInfo.clear()
    }
}

// might as well create this just once
val turnIndices = arrayOf(NDArrayIndex.point(0), NDArrayIndex.point(0))

fun GameState.toProbModelInput(): INDArray {
    val sz = gameSpec.boardSize
    val np = gameSpec.pieceCount - 1
    val input = Nd4j.zeros(1, 2 * np + 1, sz, sz)
    for (x in 0 until sz) {
        for (y in 0 until sz) {
            val p = at(x, y)
            if (p != 0) {
                val channel = if (p > 0) p else np - p
                input.putScalar(intArrayOf(0, channel, x, y), 1)
            }
        }
    }
    val turn = Nd4j.ones(sz, sz)
    if (player.eq(Player.BLACK)) turn.muli(-1)
    input.put(turnIndices, turn)
    return input
}

open class AlphaZeroMctsStrategy(val model: ComputationGraph,
                                 exploration: Double,
                                 temperature: Double) :
        VanillaMctsStrategy(exploration, temperature) {

    override fun expand(state: GameState) {
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toProbModelInput())
            sInfo.Q = outputs[0].getFloat(0, 0)
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.P = outputs[1].getFloat(intArrayOf(0, next.toPolicyIndex()))
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
        val slim = slimState(state) { i, tsr ->
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
