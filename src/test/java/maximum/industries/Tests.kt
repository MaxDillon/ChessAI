package maximum.industries

import org.amshove.kluent.shouldEqual
import org.deeplearning4j.util.ModelSerializer
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.math.sqrt

fun GameState.desc(): String {
    return toString().split(":").get(1).trim()
            .replace(" WIN", "")
            .replace(" LOSE", "")
}

class TestChess() {
    val gameSpec = loadSpec("chess")

    data class Placement(
            val p: String = "", val n: String = "", val b: String = "",
            val r: String = "", val q: String = "", val k: String = "")

    fun initBoard(white: Placement, black: Placement, whiteMove: Boolean = true): GameState {
        val gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })
        fun place(placement: Placement, sign: Int) {
            val (p, n, b, r, q, k) = placement
            arrayListOf<String>(p, n, b, r, q, k).forEachIndexed { i, pstr ->
                if (pstr.length > 0) {
                    for (j in pstr.split(",")) {
                        val x = j.first() - 'a'
                        val y = j.substring(1).toInt() - 1
                        gameBoard[y][x] = sign * (i + 1)
                    }
                }
            }
        }
        place(white, 1)
        place(black, -1)
        val player = if (whiteMove) Player.WHITE else Player.BLACK
        return GameState(gameSpec, gameBoard, player, 0, 0, 0, 0, 0, 0)
    }

    @Test
    fun modelEval() {
        val state = initBoard(white = Placement(k = "a1", p = "a6", q = "a8", b = "a4"),
                              black = Placement(k = "e8", p = "e7,f7", b = "d7", n = "c8"),
                              whiteMove = false)

        val model = ModelSerializer.restoreComputationGraph("model.chess.58000")
        val search = MonteCarloTreeSearch(
                AlphaZeroMctsStrategy(model, 1.0, 0.1), 100)
//        val search = MonteCarloTreeSearch(
//                DirichletMctsStrategy(1.0, 0.1, floatArrayOf(1f, 0f, 0.5f)), 5000)

        state.printBoard()
        val (next, slim) = search.next(state)
        next.printBoard()
        println(slim!!.player)
        for (tsr in slim!!.treeSearchResults) {
            println("${tsr.index} ${tsr.prob}")
        }
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

    @Test
    fun dirichlet() {
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(1.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(2.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(3.1f, 0.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 1.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 2.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 3.1f, 0.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 1.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 2.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(0.1f, 0.1f, 3.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(1.1f, 1.1f, 1.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
        println(expectedValuePlusSdevs(floatArrayOf(2.1f, 2.1f, 2.1f), floatArrayOf(1.0f, 0.0f, 0.5f), 4f))
    }

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

    fun initBoard(white: String = "", black: String = "", whiteMove: Boolean = true): GameState {
        val gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })
        fun place(placement: String, sign: Int) {
            for (j in placement.split(",")) {
                val x = j.first() - 'a'
                val y = j.substring(1).toInt() - 1
                gameBoard[y][x] = sign
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
                                black = "c2,b3")
                .nextMoves
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
        val dataset = maximum.industries.bak.getBatch(gameSpec, StreamInstanceReader(bis), 1)

        assert(initial.toProbModelInput().equals(dataset.features[0]))
        assert(dataset.labels[0].getInt(0) == if (final.winFor(initial)) 1 else -1)
        for (i in slim.treeSearchResults.indices) {
            val tsr = slim.treeSearchResults[i]
            assert(tsr.prob == dataset.labels[1].getFloat(0, tsr.index))
        }
    }

    @Test
    fun instanceSerialization() {
        val search = MonteCarloTreeSearch(VanillaMctsStrategy(
                1.0, 1.0), 100)

        val whiteMove = initBoard(white = "a1",
                                  black = "c1",
                                  whiteMove = false)
        val blackMove = initBoard(white = "a1,a2",
                                  black = "c1",
                                  whiteMove = false)
        val winWhite = initBoard(white = "a1,a2,a3",
                                 black = "c1,c2",
                                 whiteMove = false)
        val winBlack = initBoard(white = "a1,a2,a3",
                                 black = "c1,c2",
                                 whiteMove = false)

        val (_, slimWhite) = search.next(whiteMove)
        check(whiteMove, slimWhite, winWhite)
        check(whiteMove, slimWhite, winBlack)

        val (_, slimBlack) = search.next(blackMove)
        check(blackMove, slimBlack, winWhite)
        check(blackMove, slimBlack, winBlack)
    }

    @Test
    fun modelEval() {
        val m0 = initBoard(white = "a2", black = "", whiteMove = false)
        val search = MonteCarloTreeSearch(AlphaZeroMctsStrategy(
                ModelSerializer.restoreComputationGraph("model.tictactoe.10000"),
                exploration = 1.0, temperature = 0.1), 300)

        m0.printBoard()
        val (m1, _) = search.next(m0)
        m1.printBoard()
        val (m2, _) = search.next(m1)
        m2.printBoard()
        val (m3, _) = search.next(m2)
        m3.printBoard()
        val (m4, _) = search.next(m3)
        m4.printBoard()
        val (m5, _) = search.next(m4)
        m5.printBoard()
        val (m6, _) = search.next(m5)
        m6.printBoard()
    }
}

fun main(args: Array<String>) {
    val x = TestConnect4()
    x.check()
}

class TestConnect4() : TwoColorSetup("connect4") {
    @Test
    fun modelEval() {
        var state = initBoard(white = "c1,e1,d1",
                              black = "d2,c2",
                              whiteMove = false)
        state.printBoard()
        val model = ModelSerializer.restoreComputationGraph("model.reboot.38000")

        val search = MonteCarloTreeSearch(
                AlphaZeroMctsStrategy(model, 1.0, 0.1), 500)
        val (next, slim) = search.next(state)
        next.printBoard()
    }

    fun checkModelConsistency(path: String) {
        val model = ModelSerializer.restoreComputationGraph(path)

        var sump = 0f
        var sumv = 0f
        var sump2 = 0f
        var sumv2 = 0f
        var sumpv = 0f
        var n = 0

        for (i in 0 until 50) {
            var state = initBoard()
            while (state.outcome == Outcome.UNDETERMINED) {
                state.printBoard()
                println(state.toProbModelInput())
                val outputs = model.output(state.toProbModelInput())
                for (j in 0 until 49) {
                    println("${j / 7} ${j % 7}: ${outputs[1].getFloat(intArrayOf(0, j)).f3()} ${outputs[2].getFloat(
                            intArrayOf(0, j)).f3()}")
                }
                for (next in state.nextMoves) {
                    val nextp = outputs[1].getFloat(intArrayOf(0, next.toPolicyIndex()))
                    val nextv = model.output(next.toProbModelInput())[0].getFloat(0, 0)

                    sump += nextp
                    sumv += nextv
                    sump2 += nextp * nextp
                    sumv2 += nextv * nextv
                    sumpv += nextp * nextv
                    n += 1
                }
                state = state.nextMoves[rand.nextInt(state.nextMoves.size)]
            }
        }

        val cov = sumpv / n - (sump / n) * (sumv / n)
        val sdp = sqrt(sump2 / n - (sump / n) * (sump / n))
        val sdv = sqrt(sumv2 / n - (sumv / n) * (sumv / n))
        val correl = cov / sdp / sdv

        println("${correl.f3()}\t $path")
    }

    @Test
    fun check() {
//        checkModelConsistency("model.4191452.2000")
//        checkModelConsistency("model.4191452.4000")
//        checkModelConsistency("model.4191452.6000")
//        checkModelConsistency("model.4191452.8000")
//        checkModelConsistency("model.4191452.10000")
//        checkModelConsistency("model.4191452.12000")
//        checkModelConsistency("model.4191452.14000")
//        checkModelConsistency("model.4191452.16000")
//        checkModelConsistency("model.4191452.18000")
//        checkModelConsistency("model.4191452.20000")
        checkModelConsistency("model.4191452.22000")
        checkModelConsistency("model.4191452.24000")
//        checkModelConsistency("model.4191200.1000")
//        checkModelConsistency("model.4191200.2000")
//        checkModelConsistency("model.4191200.3000")
//        checkModelConsistency("model.4191200.4000")
//        checkModelConsistency("model.4191200.6000")
//        checkModelConsistency("model.4191200.8000")
//        checkModelConsistency("model.4191200.10000")

//        checkModelConsistency("model.4190851.2000")
//        checkModelConsistency("model.4190851.4000")
//        checkModelConsistency("model.4190851.6000")
//        checkModelConsistency("model.4190851.8000")
//        checkModelConsistency("model.4190851.10000")
//        checkModelConsistency("model.4190851.12000")
//        checkModelConsistency("model.4190851.14000")
//        checkModelConsistency("model.4190851.16000")
//        checkModelConsistency("prod_model.connect4")
//        checkModelConsistency("model.c4_4181236.10000")
//        checkModelConsistency("model.c4_4181236.20000")
//        checkModelConsistency("model.c4_4181236.30000")
//        checkModelConsistency("model.c4_4181236.40000")
//        checkModelConsistency("model.c4_4181610.10000")
//        checkModelConsistency("model.c4_4181610.20000")
//        checkModelConsistency("model.c4_4181610.30000")
//        checkModelConsistency("model.c4_4181610.40000")
//        checkModelConsistency("model.c4_4181610.50000")
//        checkModelConsistency("model.c4_4181610.80000")
//        checkModelConsistency("model.connect4_new.10000")
//        checkModelConsistency("model.connect4_new.20000")
//        checkModelConsistency("model.connect4_new.30000")
//        checkModelConsistency("model.connect4_new.50000")
//        checkModelConsistency("model.connect4_new.70000")
//        checkModelConsistency("model.reboot.10000")
    }
}

class TestINDArray() {
    @Test
    fun testArray() {
        val arr = Nd4j.zeros(2, 2, 3, 3)
        val ia = intArrayOf(0, 0)
        val onez = Nd4j.ones(3, 3)
        arr.put(arrayOf(NDArrayIndex.point(0), NDArrayIndex.point(0)), onez)
        println(arr)
    }
}

