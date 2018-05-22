package maximum.industries

import org.amshove.kluent.shouldBeGreaterThan
import org.amshove.kluent.shouldEqual
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.util.ModelSerializer
import org.junit.Test
import org.nd4j.linalg.lossfunctions.impl.PseudoSpherical
import org.nd4j.shade.jackson.databind.jsontype.NamedType
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.*
import kotlin.math.sqrt

fun GameState.desc(): String {
    return toString().split(":").get(1).trim()
            .replace(" WIN", "")
            .replace(" LOSE", "")
            .replace(" DRAW", "")
}

fun checkStatesEqual(state1: GameState, state2: GameState) {
    state1.player.shouldEqual(state2.player)
    for (x in 0 until state1.gameSpec.boardSize) {
        for (y in 0 until state1.gameSpec.boardSize) {
            state1.at(x, y).shouldEqual(state2.at(x, y))
        }
    }
}

class TestChess() {
    val gameSpec = loadSpec("chess")

    data class Placement(
            val p: String = "", val n: String = "", val b: String = "",
            val r: String = "", val q: String = "", val k: String = "")

    fun initBoard(white: Placement, black: Placement,
                  whiteMove: Boolean = true, depth: Short = 0): GameState {
        val gameBoard = ByteArray(gameSpec.boardSize * gameSpec.boardSize) { 0 }
        fun place(placement: Placement, sign: Int) {
            val (p, n, b, r, q, k) = placement
            arrayListOf(p, n, b, r, q, k).forEachIndexed { i, pstr ->
                if (pstr.length > 0) {
                    for (j in pstr.split(",")) {
                        val x = j.first() - 'a'
                        val y = j.substring(1).toInt() - 1
                        gameBoard[y * gameSpec.boardSize + x] = (sign * (i + 1)).toByte()
                    }
                }
            }
        }
        place(white, 1)
        place(black, -1)
        val player = if (whiteMove) Player.WHITE else Player.BLACK
        return GameState(gameSpec, gameBoard, player,
                         0, 0, 0, 0, 0, depth)
    }

    @Test
    fun modelEval() {
        val state = initBoard(white = Placement(k = "c8"),
                              black = Placement(k = "g4", q = "h7", b = "c6", p = "b4"),
                              whiteMove = true)

        val model1 = ModelSerializer.restoreComputationGraph("prod_model.chess")
        val output1 = model1.output(state.toModelInput())

        val model2 = ModelSerializer.restoreComputationGraph("model.zero.10000")
        val output2 = model2.output(state.toModelInput())

        val indices1 = ArrayList<MoveInfo>()
        for (i in 0 until policySize(gameSpec))
            if (output1[1].getFloat(0, i) > 0.002) indices1.add(gameSpec.expandPolicyIndex(i))

        val indices2 = ArrayList<MoveInfo>()
        for (i in 0 until policySize(gameSpec))
            if (output2[1].getFloat(0, i) > 0.002) indices2.add(gameSpec.expandPolicyIndex(i))

        val params = SearchParameters()
//        val search = MonteCarloTreeSearch(AlphaZeroMctsStrategy(model, params), params)
//
//        state.printBoard()
//        val (next, slim) = search.next(state)
//        for (tsr in slim!!.treeSearchResults) {
//            println("${gameSpec.expandPolicyIndex(tsr.index)} ${tsr.prob} ${output[1].getFloat(0, tsr.index)}")
//        }
    }

    @Test
    fun pawnMoves() {
        initBoard(white = Placement(p = "a2"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("a2 -> a3", "a2 -> a4"))

        initBoard(white = Placement(p = "a3"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("a3 -> a4"))

        initBoard(white = Placement(p = "a2,b2,c2,d2,e2,f2,g2,h2"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("a2 -> a3", "a2 -> a4", "b2 -> b3", "b2 -> b4",
                                    "c2 -> c3", "c2 -> c4", "d2 -> d3", "d2 -> d4",
                                    "e2 -> e3", "e2 -> e4", "f2 -> f3", "f2 -> f4",
                                    "g2 -> g3", "g2 -> g4", "h2 -> h3", "h2 -> h4"))

        initBoard(white = Placement(p = "a7"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("a7 -> a8"))

        initBoard(white = Placement(p = "a3"),
                  black = Placement(p = "a4"))
                .nextMoves.map { it.desc() }.size
                .shouldEqual(0)

        initBoard(white = Placement(p = "a3,a4"),
                  black = Placement(p = "a5"))
                .nextMoves.map { it.desc() }.size
                .shouldEqual(0)

        initBoard(white = Placement(p = "b3"),
                  black = Placement(p = "a4,c4"))
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b3 -> a4", "b3 -> b4", "b3 -> c4"))

        initBoard(white = Placement(p = "a4,b3,c4"),
                  black = Placement(p = "a5,c5"))
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b3 -> b4"))

        initBoard(white = Placement(p = "b2"),
                  black = Placement(p = "a3,c3"))
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b2 -> a3", "b2 -> b3", "b2 -> b4", "b2 -> c3"))

        initBoard(white = Placement(p = "a6,c6"),
                  black = Placement(p = "b7"),
                  whiteMove = false)
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b7 -> a6", "b7 -> b5", "b7 -> b6", "b7 -> c6"))

        initBoard(white = Placement(),
                  black = Placement(p = "b2"),
                  whiteMove = false)
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b2 -> b1"))
    }

    @Test
    fun knightMoves() {
        initBoard(white = Placement(n = "b1"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("b1 -> a3", "b1 -> c3", "b1 -> d2"))

        initBoard(white = Placement(n = "d4"),
                  black = Placement())
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("d4 -> b3", "d4 -> b5", "d4 -> c2", "d4 -> c6",
                                    "d4 -> e2", "d4 -> e6", "d4 -> f3", "d4 -> f5"))
    }

    @Test
    fun testConversions() {
        // GameState => ModelInput => GameState (white move)
        val state1a = initBoard(white = Placement(k = "d1", n = "c5", r = "h7", p = "b2,c3,d2"),
                                black = Placement(k = "f7", p = "f6"),
                                whiteMove = true)
        val state1b = gameSpec.fromModelInput(state1a.toModelInput(), 0)
        checkStatesEqual(state1a, state1b)

        // GameState => ModelInput => GameState (black move)
        val state2a = state1a.nextMoves[0]
        val state2b = gameSpec.fromModelInput(state2a.toModelInput(), 0)
        checkStatesEqual(state2a, state2b)

        // GameState => Slim => Instance => BatchInput => GameState
        val slim = state1a.toSlimState { _, tsr -> tsr.prob = 0.5f }
        val white_win = initBoard(white = Placement(k = "a1"), black = Placement(),
                                  whiteMove = false, depth = 10)
        val white_loss = initBoard(black = Placement(k = "a1"), white = Placement(),
                                   whiteMove = true, depth = 20)
        val instance_win = slim.toTrainingInstance(white_win)
        val instance_loss = slim.toTrainingInstance(white_loss)
        val (input, value, policy, legal) = initTrainingData(gameSpec, 2)
        instance_win.toBatchTrainingInput(gameSpec, 0, 0, input, value, policy, legal)
        instance_loss.toBatchTrainingInput(gameSpec, 1, 0, input, value, policy, legal)
        val state1c = gameSpec.fromModelInput(input, 0)
        val state1d = gameSpec.fromModelInput(input, 1)

        // check transported states are equal
        checkStatesEqual(state1a, state1c)
        checkStatesEqual(state1a, state1d)

        // check transport of win/loss info
        state1a.player.shouldEqual(Player.WHITE)
        slim.player.shouldEqual(Instance.Player.WHITE)
        white_win.player.shouldEqual(Player.BLACK)
        white_loss.player.shouldEqual(Player.WHITE)
        white_win.outcome.shouldEqual(Outcome.LOSE)
        white_loss.outcome.shouldEqual(Outcome.LOSE)
        state1c.player.shouldEqual(Player.WHITE)
        state1d.player.shouldEqual(Player.WHITE)
        instance_win.player.shouldEqual(Instance.Player.WHITE)
        instance_loss.player.shouldEqual(Instance.Player.WHITE)
        value.getFloat(0).shouldEqual(1f)
        value.getFloat(1).shouldEqual(-1f)

        // check transport of game length
        instance_win.gameLength.shouldEqual(white_win.moveDepth.toInt())
        instance_loss.gameLength.shouldEqual(white_loss.moveDepth.toInt())
    }

    @Test
    fun testReflections() {
        // white move to/from model input
        val state1 = initBoard(white = Placement(k = "d1", n = "b3"),
                               black = Placement(k = "g7"), whiteMove = true)
        val refl_1 = initBoard(white = Placement(k = "e1", n = "g3"),
                               black = Placement(k = "b7"), whiteMove = true)
        val refl_2 = initBoard(black = Placement(k = "d8", n = "b6"),
                               white = Placement(k = "g2"), whiteMove = false)
        val refl_3 = initBoard(black = Placement(k = "e8", n = "g6"),
                               white = Placement(k = "b2"), whiteMove = false)

        // set up slim / instance / batch test
        val (input, value, policy, legal) = initTrainingData(gameSpec, 4)

        val white_win = initBoard(white = Placement(k = "a1"), black = Placement(), whiteMove = false)
        val slim = state1.toSlimState { _, tsr -> tsr.prob = 0.5f }
        val instance_win = slim.toTrainingInstance(white_win)

        instance_win.toBatchTrainingInput(gameSpec, 0, 0, input, value, policy, legal)
        instance_win.toBatchTrainingInput(gameSpec, 1, 1, input, value, policy, legal)
        instance_win.toBatchTrainingInput(gameSpec, 2, 2, input, value, policy, legal)
        instance_win.toBatchTrainingInput(gameSpec, 3, 3, input, value, policy, legal)

        val state2 = gameSpec.fromModelInput(input, 0)
        val state3 = gameSpec.fromModelInput(input, 1)
        val state4 = gameSpec.fromModelInput(input, 2)
        val state5 = gameSpec.fromModelInput(input, 3)

        checkStatesEqual(state2, state1)
        checkStatesEqual(state3, refl_1)
        checkStatesEqual(state4, refl_2)
        checkStatesEqual(state5, refl_3)
    }

    @Test
    fun testLegalAndPolicy() {
        val state1 = initBoard(white = Placement(k = "d1", b = "a2,e1", r = "b1"),
                               black = Placement(k = "h8", p = "g7,h7,g5", b = "h6"), whiteMove = false)
        val final = initBoard(white = Placement(k = "d1", b = "a2,h8", r = "b8"),
                              black = Placement(p = "g6,h6,g5", b = "f8"), whiteMove = false)
        val params = SearchParameters(exploration = 0.2, temperature = 0.01, iterations = 5000)
        val search = MonteCarloTreeSearch(VanillaMctsStrategy(params), params)
        val (_, slim) = search.next(state1)
        val instance = slim!!.toTrainingInstance(final)

        val (input, value, policy, legal) = initTrainingData(gameSpec, 4)
        instance.toBatchTrainingInput(gameSpec, 0, 0, input, value, policy, legal)
        instance.toBatchTrainingInput(gameSpec, 1, 1, input, value, policy, legal)
        instance.toBatchTrainingInput(gameSpec, 2, 2, input, value, policy, legal)
        instance.toBatchTrainingInput(gameSpec, 3, 3, input, value, policy, legal)

        fun checkLegalAndPolicy(instance: Int, moves: IntArray) {
            for (i in 0 until policySize(gameSpec)) {
                val isLegal = moves.contains(i)
                legal.getFloat(intArrayOf(instance, i)).shouldEqual(if (isLegal) 1f else -1f)
                if (isLegal) {
                    policy.getFloat(intArrayOf(instance, i)).shouldBeGreaterThan(0f)
                } else {
                    policy.getFloat(intArrayOf(instance, i)).shouldEqual(0f)
                }
            }
        }

        // Three legal moves:
        val g5g4 = gameSpec.toPolicyIndex(MoveInfo(6, 4, 6, 3, 0))
        val g7g6 = gameSpec.toPolicyIndex(MoveInfo(6, 6, 6, 5, 0))
        val h8g8 = gameSpec.toPolicyIndex(MoveInfo(7, 7, 6, 7, 0))
        checkLegalAndPolicy(0, intArrayOf(g5g4, g7g6, h8g8))

        val b5b4 = gameSpec.toPolicyIndex(MoveInfo(1, 4, 1, 3, 0))
        val b7b6 = gameSpec.toPolicyIndex(MoveInfo(1, 6, 1, 5, 0))
        val a8b8 = gameSpec.toPolicyIndex(MoveInfo(0, 7, 1, 7, 0))
        checkLegalAndPolicy(1, intArrayOf(b5b4, b7b6, a8b8))

        val h1g1 = gameSpec.toPolicyIndex(MoveInfo(7, 0, 6, 0, 0))
        val g2g3 = gameSpec.toPolicyIndex(MoveInfo(6, 1, 6, 2, 0))
        val g4g5 = gameSpec.toPolicyIndex(MoveInfo(6, 3, 6, 4, 0))
        checkLegalAndPolicy(2, intArrayOf(h1g1, g2g3, g4g5))

        val a1b1 = gameSpec.toPolicyIndex(MoveInfo(0, 0, 1, 0, 0))
        val b2b3 = gameSpec.toPolicyIndex(MoveInfo(1, 1, 1, 2, 0))
        val b4b5 = gameSpec.toPolicyIndex(MoveInfo(1, 3, 1, 4, 0))
        checkLegalAndPolicy(3, intArrayOf(a1b1, b2b3, b4b5))
    }

    @Test
    fun testThreefoldRepetition() {
        var state = GameState(gameSpec)
        state = state.nextMoves.filter { it.desc() == "b1 -> a3" }.first()
        state = state.nextMoves.filter { it.desc() == "b8 -> a6" }.first()
        state = state.nextMoves.filter { it.desc() == "a3 -> b1" }.first()
        state = state.nextMoves.filter { it.desc() == "a6 -> b8" }.first()
        state = state.nextMoves.filter { it.desc() == "b1 -> a3" }.first()
        state = state.nextMoves.filter { it.desc() == "b8 -> a6" }.first()
        state = state.nextMoves.filter { it.desc() == "a3 -> b1" }.first()
        state = state.nextMoves.filter { it.desc() == "a6 -> b8" }.first()
        state.outcome.shouldEqual(Outcome.DRAW)
    }

    fun expectedValuePlusSdevs(wld: FloatArray, values: FloatArray, sdevs: Float): Float {
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

//    @Test
//    fun dirichlet() {
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(1.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(2.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(3.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 1.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 2.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 3.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 1.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 2.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 3.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(1.1f, 1.1f, 1.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//        println(expectedValuePlusSdevs(floatArrayOf(2.1f, 2.1f, 2.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
//    }

//    @Test
//    fun searchTest() {
//        val state = initBoard(
//                white = Placement(
//                        p = "a2,b2,c2,d2,f1,f2",
//                        r = "a1,d1",
//                        n = "f5",
//                        k = "e1"
//                ),
//                black = Placement(
//                        q = "e6",
//                        k = "c8",
//                        r = "h8",
//                        p = "a7,b7,f7,g7",
//                        n = "c6"
//                ))
//        treeSearchSelfValue(state, {})
//    }
//
//    @Test
//    fun moveIndexTest() {
//        initBoard(
//                white = Placement(p = "c5"),
//                black = Placement(p = "c6,b6", k = "d6"))
//                .nextMoves.map { it.getMoveIndex() }.sorted()
//                .shouldEqual(listOf((4 + 2 * 8) * (8 * 8) + (5 + 1 * 8),
//                                    (4 + 2 * 8) * (8 * 8) + (5 + 3 * 8)))
//    }
}

open class TwoColorSetup(name: String) {
    val gameSpec = loadSpec(name)

    init {
        NeuralNetConfiguration.reinitMapperWithSubtypes(
                Collections.singletonList(NamedType(PseudoSpherical::class.java)))
    }

    fun initBoard(white: String = "", black: String = "", whiteMove: Boolean = true): GameState {
        val gameBoard = ByteArray(gameSpec.boardSize * gameSpec.boardSize) { 0 }
        fun place(placement: String, sign: Int) {
            for (j in placement.split(",")) {
                val x = j.first() - 'a'
                val y = j.substring(1).toInt() - 1
                gameBoard[y * gameSpec.boardSize + x] = sign.toByte()
            }
        }
        if (white != "") place(white, 1)
        if (black != "") place(black, -1)
        val player = if (whiteMove) Player.WHITE else Player.BLACK
        return GameState(gameSpec, gameBoard, player, 0, 0, 0, 0, 0, 0)
    }
}

class TestOthello() : TwoColorSetup("othello") {
    @Test
    fun moves() {
        initBoard(white = "d4,e5",
                  black = "d5,e4")
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("d4 -> d6", "d4 -> f4", "e5 -> c5", "e5 -> e3"))

        initBoard(white = "a1,a2,b1",
                  black = "a3,b2,c1")
                .nextMoves.map { it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> c3", "a2 -> a4", "a2 -> c2", "b1 -> b3", "b1 -> d1"))

        initBoard(white = "a1,a2,b1",
                  black = "a3,b2,c1",
                  whiteMove = false)
                .nextMoves.map { it.desc() }.size
                .shouldEqual(0)

        // Test chaining, exchanges, impressment
        // Impress one piece of the other, exchange for P2
        val states1 = initBoard(white = "c1,a3",
                                black = "c2,b3").nextMoves
        states1.map { it.desc() }.sorted().shouldEqual(listOf("a3 -> c3", "c1 -> c3"))
        (states1[0].at(2, 1) * states1[0].at(1, 2)).shouldEqual(-1)
        (states1[1].at(2, 1) * states1[1].at(1, 2)).shouldEqual(-1)
        (states1[0].at(2, 1) * states1[1].at(2, 1)).shouldEqual(-1)
        states1[0].at(2, 2).shouldEqual(2)
        states1[1].at(2, 2).shouldEqual(2)
        states1[0].player.shouldEqual(Player.WHITE)
        states1[1].player.shouldEqual(Player.WHITE)

        // P2 does flips in second direction
        val states2 = states1[0].nextMoves
        states2.size.shouldEqual(1)
        states2[0].at(1, 2).shouldEqual(1)
        states2[0].at(2, 1).shouldEqual(1)
        states2[0].at(2, 2).shouldEqual(2)
        states2[0].player.shouldEqual(Player.WHITE)

        // P2 does flips in second direction
        val states3 = states1[1].nextMoves
        states3.size.shouldEqual(1)
        states3[0].at(1, 2).shouldEqual(1)
        states3[0].at(2, 1).shouldEqual(1)
        states3[0].at(2, 2).shouldEqual(2)
        states3[0].player.shouldEqual(Player.WHITE)

        // Back to P1 and pass
        val states4 = states2[0].nextMoves
        states4.size.shouldEqual(1)
        states4[0].at(2, 2).shouldEqual(1)
        states4[0].player.shouldEqual(Player.BLACK)

        // Back to P1 and pass
        val states5 = states3[0].nextMoves
        states5.size.shouldEqual(1)
        states5[0].at(2, 2).shouldEqual(1)
        states5[0].player.shouldEqual(Player.BLACK)
    }
}

class TestTicTacToe() : TwoColorSetup("tictactoe") {
    fun check(initial: GameState, slim: SlimState?, final: GameState) {
        val bos = ByteArrayOutputStream()
        recordGame(final, arrayListOf(slim!!), bos)
        val bis = ByteArrayInputStream(bos.toByteArray())
        val dataset = maximum.industries.getBatch(gameSpec, StreamInstanceReader(bis),
                                                  useValue = true, usePolicy = true, useLegal = false,
                                                  batchSize = 1)

        assert(initial.toModelInput().equals(dataset.features[0]))
        assert(dataset.labels[0].getInt(0) == if (final.winFor(initial)) 1 else -1)
        for (i in slim.treeSearchResults.indices) {
            val tsr = slim.treeSearchResults[i]
            assert(tsr.prob == dataset.labels[1].getFloat(0, tsr.index))
        }
    }

    // This is tested more thoroughly in the chess test. And this is broken here due to reflections.

//    @Test
//    fun instanceSerialization() {
//        val search = MonteCarloTreeSearch(VanillaMctsStrategy(
//                1.0, 1.0), 100)
//
//        val whiteMove = initBoard(white = "a1",
//                                  black = "c1",
//                                  whiteMove = false)
//        val blackMove = initBoard(white = "a1,a2",
//                                  black = "c1",
//                                  whiteMove = false)
//        val winWhite = initBoard(white = "a1,a2,a3",
//                                 black = "c1,c2",
//                                 whiteMove = false)
//        val winBlack = initBoard(white = "a1,a2,a3",
//                                 black = "c1,c2",
//                                 whiteMove = false)
//
//        val (_, slimWhite) = search.next(whiteMove)
//        check(whiteMove, slimWhite, winWhite)
//        check(whiteMove, slimWhite, winBlack)
//
//        val (_, slimBlack) = search.next(blackMove)
//        check(blackMove, slimBlack, winWhite)
//        check(blackMove, slimBlack, winBlack)
//    }
}
