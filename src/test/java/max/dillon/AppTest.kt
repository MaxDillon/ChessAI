package max.dillon

import org.amshove.kluent.shouldEqual
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class TestChess() {
    val gameSpec = loadSpec("chess")

    data class Placement(
            val p: String = "", val n: String = "", val b: String = "",
            val r: String = "", val q: String = "", val k: String = "")

    fun initBoard(white: Placement, black: Placement, whiteMove: Boolean = true,
                  model: ComputationGraph? = null): GameState {
        val state = GameState(gameSpec, model)
        state.whiteMove = whiteMove
        state.gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })

        fun place(placement: Placement, sign: Int) {
            val (p, n, b, r, q, k) = placement
            arrayListOf<String>(p, n, b, r, q, k).forEachIndexed { i, pstr ->
                if (pstr.length > 0) {
                    for (j in pstr.split(",")) {
                        val x = j.first() - 'a'
                        val y = j.substring(1).toInt() - 1
                        state.gameBoard[y][x] = sign * (i + 1)
                    }
                }
            }
        }
        place(white, 1)
        place(black, -1)
        return state
    }

    @Test
    fun pawnMoves() {
        initBoard(white = Placement(p = "a2"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("a2 -> a3", "a2 -> a4"))

        initBoard(white = Placement(p = "a3"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("a3 -> a4"))

        initBoard(white = Placement(p = "a2,b2,c2,d2,e2,f2,g2,h2"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("a2 -> a3", "a2 -> a4", "b2 -> b3", "b2 -> b4",
                                    "c2 -> c3", "c2 -> c4", "d2 -> d3", "d2 -> d4",
                                    "e2 -> e3", "e2 -> e4", "f2 -> f3", "f2 -> f4",
                                    "g2 -> g3", "g2 -> g4", "h2 -> h3", "h2 -> h4"))

        initBoard(white = Placement(p = "a7"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("a7 -> a8"))

        initBoard(white = Placement(p = "a3"),
                  black = Placement(p = "a4"))
                .getLegalNextStates().map { it.description }.size
                .shouldEqual(0)

        initBoard(white = Placement(p = "a3,a4"),
                  black = Placement(p = "a5"))
                .getLegalNextStates().map { it.description }.size
                .shouldEqual(0)

        initBoard(white = Placement(p = "b3"),
                  black = Placement(p = "a4,c4"))
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b3 -> a4", "b3 -> b4", "b3 -> c4"))

        initBoard(white = Placement(p = "a4,b3,c4"),
                  black = Placement(p = "a5,c5"))
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b3 -> b4"))

        initBoard(white = Placement(p = "b2"),
                  black = Placement(p = "a3,c3"))
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b2 -> a3", "b2 -> b3", "b2 -> b4", "b2 -> c3"))

        initBoard(white = Placement(p = "a6,c6"),
                  black = Placement(p = "b7"),
                  whiteMove = false)
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b7 -> a6", "b7 -> b5", "b7 -> b6", "b7 -> c6"))

        initBoard(white = Placement(),
                  black = Placement(p = "b2"),
                  whiteMove = false)
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b2 -> b1"))
    }

    @Test
    fun knightMoves() {
        initBoard(white = Placement(n = "b1"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("b1 -> a3", "b1 -> c3", "b1 -> d2"))

        initBoard(white = Placement(n = "d4"),
                  black = Placement())
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("d4 -> b3", "d4 -> b5", "d4 -> c2", "d4 -> c6",
                                    "d4 -> e2", "d4 -> e6", "d4 -> f3", "d4 -> f5"))
    }

    @Test
    fun searchTest() {
        val state = initBoard(
                white = Placement(
                        p = "a2,b2,c2,d2,f1,f2",
                        r = "a1,d1",
                        n = "f5",
                        k = "e1"
                ),
                black = Placement(
                        q = "e6",
                        k = "c8",
                        r = "h8",
                        p = "a7,b7,f7,g7",
                        n = "c6"
                ))
        treeSearchSelfValue(state, {})
    }

    @Test
    fun moveIndexTest() {
        initBoard(
                white = Placement(p = "c5"),
                black = Placement(p = "c6,b6", k = "d6"))
                .nextMoves.map { it.getMoveIndex() }.sorted()
                .shouldEqual(listOf((4 + 2 * 8) * (8 * 8) + (5 + 1 * 8),
                                    (4 + 2 * 8) * (8 * 8) + (5 + 3 * 8)))
    }
}

open class TwoColorSetup(name: String) {
    val gameSpec = loadSpec(name)

    fun initBoard(white: String, black: String, whiteMove: Boolean = true,
                  model: ComputationGraph? = null): GameState {
        val state = GameState(gameSpec, model)
        state.whiteMove = whiteMove
        state.gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })

        fun place(placement: String, sign: Int) {
            for (j in placement.split(",")) {
                val x = j.first() - 'a'
                val y = j.substring(1).toInt() - 1
                state.gameBoard[y][x] = sign
            }
        }
        place(white, 1)
        place(black, -1)
        return state
    }
}

class TestOthello() : TwoColorSetup("othello") {
    @Test
    fun moves() {
        initBoard(white = "d4,e5",
                  black = "d5,e4")
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("d4 -> d6", "d4 -> f4", "e5 -> c5", "e5 -> e3"))

        initBoard(white = "a1,a2,b1",
                  black = "a3,b2,c1")
                .getLegalNextStates().map { it.description }.sorted()
                .shouldEqual(listOf("a1 -> c3", "a2 -> a4", "a2 -> c2", "b1 -> b3", "b1 -> d1"))

        initBoard(white = "a1,a2,b1",
                  black = "a3,b2,c1",
                  whiteMove = false)
                .getLegalNextStates().map { it.description }.size
                .shouldEqual(0)

        // Test chaining, exchanges, impressment
        // Impress one piece of the other, exchange for P2
        val states1 = initBoard(white = "c1,a3",
                                black = "c2,b3")
                .getLegalNextStates()
        states1.map { it.description }.sorted().shouldEqual(listOf("a3 -> c3", "c1 -> c3"))
        (states1[0].at(2, 1) * states1[0].at(1, 2)).shouldEqual(-1)
        (states1[1].at(2, 1) * states1[1].at(1, 2)).shouldEqual(-1)
        (states1[0].at(2, 1) * states1[1].at(2, 1)).shouldEqual(-1)
        states1[0].at(2, 2).shouldEqual(2)
        states1[1].at(2, 2).shouldEqual(2)
        states1[0].whiteMove.shouldEqual(true)
        states1[1].whiteMove.shouldEqual(true)

        // P2 does flips in second direction
        val states2 = states1[0].getLegalNextStates()
        states2.size.shouldEqual(1)
        states2[0].at(1, 2).shouldEqual(1)
        states2[0].at(2, 1).shouldEqual(1)
        states2[0].at(2, 2).shouldEqual(2)
        states2[0].whiteMove.shouldEqual(true)

        // P2 does flips in second direction
        val states3 = states1[1].getLegalNextStates()
        states3.size.shouldEqual(1)
        states3[0].at(1, 2).shouldEqual(1)
        states3[0].at(2, 1).shouldEqual(1)
        states3[0].at(2, 2).shouldEqual(2)
        states3[0].whiteMove.shouldEqual(true)

        // Back to P1 and pass
        val states4 = states2[0].getLegalNextStates()
        states4.size.shouldEqual(1)
        states4[0].at(2, 2).shouldEqual(1)
        states4[0].whiteMove.shouldEqual(false)

        // Back to P1 and pass
        val states5 = states3[0].getLegalNextStates()
        states5.size.shouldEqual(1)
        states5[0].at(2, 2).shouldEqual(1)
        states5[0].whiteMove.shouldEqual(false)
    }
}


class TestTicTacToe() : TwoColorSetup("tictactoe") {
    @Test
    fun instanceSerialization() {
        val current = initBoard(white = "a2,a3", black = "b2", whiteMove = false)
        val final = initBoard(white = "a1,a2,a3", black = "b1,b2", whiteMove = false)
        treeSearchMove(current, 1.0)

        val bos = ByteArrayOutputStream()
        recordGame(final, arrayListOf(slim(current)), bos)

        val bis = ByteArrayInputStream(bos.toByteArray())
        val dataset = getBatch(gameSpec, StreamInstanceReader(bis), 1)

        println("toModelInput")
        println(current.toModelInput())
        println("serizlized/deserialized")
        println(dataset.features)
        println("pHat")
        println(dataset.labels)
    }

    @Test
    fun modelEval() {
        val m0 = initBoard(
                white = "b1,c3", black = "a1,c1", whiteMove = true,
                model = ModelSerializer.restoreComputationGraph("model.tictactoe.3000"))
        println(m0.whiteMove)
        m0.printBoard()

        val m1 = treeSearchMove(m0, 0.5)
        println(m0.value)
        for (next in m0.nextMoves) println("${next.description} ${next.prior} ${next.pi}")
        println(m1.whiteMove)
        m1.printBoard()

        val m2 = treeSearchMove(m1, 0.5)
        println(m1.value)
        for (next in m1.nextMoves) println("${next.description} ${next.prior} ${next.pi}")
        println(m2.whiteMove)
        m2.printBoard()

        val m3 = treeSearchMove(m2, 0.5)
        println(m2.value)
        for (next in m2.nextMoves) println("${next.description} ${next.prior} ${next.pi}")
        println(m3.whiteMove)
        m3.printBoard()

        println(m3.predict().first)
    }

}

//class TestConnect4() : TwoColorSetup("connect4") {
//    @Test
//    fun modelEval() {
//        val state = initBoard(white="c1,d1,d2,d3,f1,e2,g1",
//                              black="c2,e1,a1,d4,g1,e3,b1",
//                              whiteMove = true,
//                              model=ModelSerializer.restoreComputationGraph("model.connect4.3940"))
//        state.printBoard()
//        println(state.toModelInput())
//
//        treeSearchMove(state, 0.5)
//        for (next in state.nextMoves) {
//            println("${next.description} ${next.prior} ${next.pi}")
//        }
//    }
//}



