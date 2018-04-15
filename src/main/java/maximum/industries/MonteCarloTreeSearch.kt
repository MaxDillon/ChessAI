package maximum.industries

import maximum.industries.GameGrammar.MoveSource
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.lang.Math.pow
import java.util.*
import kotlin.math.abs
import kotlin.math.sqrt

val rand = Random(System.currentTimeMillis())

interface GameSearchAlgo {
    fun next(state: GameState): Pair<GameState, SlimState?>
}

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
    // rollout or model eval) and whose children, if any, have priors and initialized
    // counts for terminal nodes. The algorithm will expand one node per iteration.
    fun expanded(state: GameState): Boolean

    // Returns a priority balancing three components:
    // (1) the current estimated value of the node (+/-1 for terminal wins/losses)
    // (2) the information value of exploring the node (zero for terminal nodes)
    // (3) the bias value of an immediate terminal win
    fun searchPriority(s1: GameState, s2: GameState): Double

    // For a terminal node just increments visit count and sets fixed value. For a
    // non-terminal node (1) initializes child priors, (2) sets child values and
    // priming visit counts for any terminal child nodes, (3) sets initial node value
    // via rollout or model. During rollout immediate winning moves must be taken.
    fun expand(state: GameState)

    // Backprop estimated values of just-expanded nodes to the root of the search.
    fun backprop(stack: List<GameState>, expanded: GameState)

    // Picks a move according to accumulated mcts stats & prunes cache
    fun pickMove(state: GameState): Pair<GameState, SlimState>
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
        return if (s1.player == s2.player) 1 else -1
    }

    fun GameState.winFor(parent: GameState): Boolean {
        return if (player == parent.player) outcome == Outcome.WIN else outcome == Outcome.LOSE
    }

    fun GameState.lossFor(parent: GameState): Boolean {
        return if (player == parent.player) outcome == Outcome.LOSE else outcome == Outcome.WIN
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
        val priorValue = s2Info.P / sqrt(1.0 + s2Info.N) // TODO: tuning
        val nodeValue = sign(s1, s2) * s2Info.Q
        val winValue = if (s2.winFor(s1)) 100.0 else 0.0 // always take immed. wins
        val infoValue =
                if (s2.outcome != Outcome.UNDETERMINED) 0.0
                else exploration * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
        return priorValue + nodeValue + winValue + infoValue
    }

    // picks a child node, first taking a win if one exists, then taking non-loss
    // moves according to the given probabilities, then taking a loss if necessary.
    fun pickChildByProbs(state: GameState, probs: DoubleArray): GameState {
        val wins = state.nextMoves.filter { it.winFor(state) }
        if (wins.isNotEmpty()) {
            // if there's a terminal win choose one
            return wins[rand.nextInt(wins.size)]
        } else {
            val sz = state.nextMoves.size
            val nonLossProbs = DoubleArray(sz) {
                if (state.nextMoves[it].lossFor(state)) 0.0 else probs[it]
            }
            val nonLossCumls = nonLossProbs.clone()
            for (i in 1 until sz) nonLossCumls[i] += nonLossCumls[i - 1]
            if (nonLossCumls[sz - 1] == 0.0) {
                // there are no non-loss moves. return a random loss.
                return state.nextMoves[rand.nextInt(sz)]
            }
            for (i in 0 until sz) {
                nonLossCumls[i] /= nonLossCumls[sz - 1]
            }
            val point = rand.nextFloat()
            return state.nextMoves[nonLossCumls.filter { it < point }.count()]
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
                rollout = pickChildByProbs(rollout, DoubleArray(rollout.nextMoves.size) { 0.0 })
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
        val player = if (state.player == Player.WHITE) {
            Instance.Player.WHITE
        } else {
            Instance.Player.BLACK
        }
        return SlimState(arr, player, tsrs)
    }

    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        for (next in state.nextMoves) {
            val nInfo = info(next)
            println("$next:\t${nInfo.N}\t${nInfo.Q.f3()}\t${nInfo.P.f3()}")
        }
        val sz = state.nextMoves.size
        val probs = DoubleArray(sz) {
            pow(info(state.nextMoves[it]).N.toDouble(), 1 / temperature)
        }
        val next = pickChildByProbs(state, probs)
        val probsum = probs.sum()
        for (i in 0 until sz) {
            probs[i] /= probsum
        }
        val slim = slimState(state) { i, tsr -> tsr.prob = probs[i].toFloat() }
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        return Pair(next, slim)
    }
}

// might as well create this just once
val turnIndices = arrayOf(NDArrayIndex.point(0), NDArrayIndex.point(0))

open class AlphaZeroMctsStrategy(val model: ComputationGraph,
                                 exploration: Double,
                                 temperature: Double) :
        VanillaMctsStrategy(exploration, temperature) {

    override fun expand(state: GameState) {
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toModelInput())
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

    fun GameState.toModelInput(): INDArray {
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
        if (player == Player.BLACK) turn.muli(-1)
        input.put(turnIndices, turn)
        return input
    }
}

class DirichletMctsStrategy(model: ComputationGraph,
                            exploration: Double,
                            temperature: Double,
                            val values: FloatArray) :
        AlphaZeroMctsStrategy(model, exploration, temperature) {

    open class DirichletInfo : Info() {
        var wld = FloatArray(3) { 0f }
    }

    override fun info(state: GameState): DirichletInfo {
        val depthInfo = allInfo.getOrPut(state.moveDepth) { HashMap() }
        return depthInfo.getOrPut(state) { DirichletInfo() } as DirichletInfo
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
        val s1Info = info(s1)
        val s2Info = info(s2)
        val nodeValue = sign(s1, s2) * s2Info.Q
        val winValue = if (s2.winFor(s1)) 100.0 else 0.0 // always take immed. wins
        val infoValue =
                if (s2.outcome != Outcome.UNDETERMINED) 0.0
                else exploration * sqrt(s1Info.N.toDouble()) / (1 + s2Info.N)
        return nodeValue + winValue + infoValue
    }

    override fun expand(state: GameState) {
        val sInfo = info(state)
        if (state.outcome == Outcome.UNDETERMINED) {
            val outputs = model.output(state.toModelInput())
            sInfo.wld[0] = outputs[0].getFloat(0, 0)
            sInfo.wld[1] = outputs[0].getFloat(0, 1)
            sInfo.wld[2] = outputs[0].getFloat(0, 2)
            for (next in state.nextMoves) {
                val nInfo = info(next)
                nInfo.Q = next.initialSelfValue()
                nInfo.wld[0] = outputs[1].getFloat(intArrayOf(0, 0, next.toPolicyIndex(), 0))
                nInfo.wld[1] = outputs[1].getFloat(intArrayOf(0, 1, next.toPolicyIndex(), 0))
                nInfo.wld[2] = outputs[1].getFloat(intArrayOf(0, 2, next.toPolicyIndex(), 0))
            }
        }
        sInfo.N += 1
        sInfo.expanded = true
    }

    override fun pickMove(state: GameState): Pair<GameState, SlimState> {
        for (next in state.nextMoves) {
            val nInfo = info(next)
            println("$next:\t${nInfo.N}\t${nInfo.wld[0].f3()}\t${nInfo.wld[1].f3()}\t${nInfo.wld[2].f3()}")
        }
        val sz = state.nextMoves.size
        val probs = DoubleArray(sz) {
            pow(info(state.nextMoves[it]).N.toDouble(), 1 / temperature)
        }
        val next = pickChildByProbs(state, probs)
        val probsum = probs.sum()
        for (i in 0 until sz) {
            probs[i] /= probsum
        }
        val slim = slimState(state) { i, tsr -> tsr.prob = probs[i].toFloat() }
        allInfo.remove(state.moveDepth) // won't be needing these anymore
        return Pair(next, slim)
    }
}
