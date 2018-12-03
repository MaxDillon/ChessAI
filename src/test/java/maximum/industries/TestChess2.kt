package maximum.industries

import maximum.industries.games.ChessState
import org.amshove.kluent.shouldBeGreaterThan
import org.amshove.kluent.shouldEqual
import org.deeplearning4j.util.ModelSerializer
import org.junit.Test

class TestChess2() {
    val gameSpec = loadSpec("chess2")

    data class Placement(
            val p: String = "", val n: String = "", val b: String = "",
            val r: String = "", val q: String = "", val k: String = "",
            val ur: String = "", val uk: String = "")

    fun initBoard(white: Placement, black: Placement,
                  whiteMove: Boolean = true, depth: Short = 0): ChessState {
        val gameBoard = ByteArray(gameSpec.boardSize * gameSpec.boardSize) { 0 }
        fun place(placement: Placement, sign: Int) {
            val (p, n, b, r, q, k, ur, uk) = placement
            arrayListOf(p, n, b, r, q, k, ur, uk).forEachIndexed { i, pstr ->
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
        return ChessState(gameSpec, gameBoard, player,
                          0, 0, 0, 0, 0, depth)
    }

    @Test
    fun modelEval() {
        // for recent self-play models:
        //   for black values whack
        //   for white values tight
        // but for old mcts models black values make sense.
        // wtf.
        val state_b0 = initBoard(white = Placement(k = "a2", p = "c4,b2,d3", n = "h1", b = "f3"),
                                 black = Placement(k = "b7", b = "c6,h4", p = "c5,h2,f7", r = "f8"), whiteMove = false)
        val state_w0 = initBoard(black = Placement(k = "a7", p = "c5,b7,d6", n = "h8", b = "f6"),
                                 white = Placement(k = "b2", b = "c3,h5", p = "c4,h7,f2", r = "f1"), whiteMove = true)

        val model = ModelSerializer.restoreComputationGraph("prod_model.chess2")
        val params = SearchParameters(exploration = 1.0, temperature = 0.1, iterations = 2500)
        val search = MonteCarloTreeSearch(AlphaZeroMctsStrategy1(model, params), params)

        val (state_b1, _) = search.next(state_b0)
        state_b0.printBoard()
        state_b1.printBoard()

        val (state_w1, _) = search.next(state_w0)
        state_w0.printBoard()
        state_w1.printBoard()
    }

    @Test
    fun wtfEval() {
        val state = initBoard(white = Placement(k = "a2", p = "a3,b2,c2,d3", r="b7,h7", q="e4"),
                              black = Placement(k = "b8", r = "g8,h6", n="g4", p="a6,e5", q="b6"), whiteMove = false)

        val model = ModelSerializer.restoreComputationGraph("prod_model.chess2")
        val params = SearchParameters(exploration = 0.1, temperature = 0.3, iterations = 200)
        val search = MonteCarloTreeSearch(AlphaZeroMctsStrategy2(model, params), params)

        val (state2, _) = search.next(state)
        state.printBoard()
        state2.printBoard()
    }

    @Test
    fun ensembleEval() {
        val state = initBoard(white = Placement(k = "a2", p = "c4,b2,d3", n = "h1", b = "f3"),
                              black = Placement(k = "b7", b = "c6,h4", p = "c5,h2,f7", r = "f8"),
                              whiteMove = false)
        val state_0 = state.gameSpec.fromModelInput(state.toModelInput(intArrayOf(0)))
        val state_1 = state.gameSpec.fromModelInput(state.toModelInput(intArrayOf(1)))
        val state_2 = state.gameSpec.fromModelInput(state.toModelInput(intArrayOf(2)))
        val state_3 = state.gameSpec.fromModelInput(state.toModelInput(intArrayOf(3)))

        val model = ModelSerializer.restoreComputationGraph("prod_model.chess2")
        val params = SearchParameters(exploration = 1.0, temperature = 0.1, iterations = 1)

        val search1 = MonteCarloTreeSearch(AlphaZeroMctsStrategy1(model, params), params)
        val search2 = MonteCarloTreeSearch(AlphaZeroMctsStrategy2(model, params), params)

        search1.next(state_0)
        search1.next(state_1)
        search1.next(state_2)
        search1.next(state_3)
        search2.next(state)
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
    fun testCheck() {
        initBoard(white = Placement(k = "d3"), black = Placement(p = "e4"), whiteMove = true)
                .inCheck().shouldEqual(true)

        initBoard(white = Placement(k = "d3"), black = Placement(p = "d4"), whiteMove = true)
                .inCheck().shouldEqual(false)

        initBoard(white = Placement(k = "d3"), black = Placement(p = "f5"), whiteMove = true)
                .inCheck().shouldEqual(false)

        initBoard(white = Placement(k = "d3"), black = Placement(r = "d8"), whiteMove = true)
                .inCheck().shouldEqual(true)

        initBoard(white = Placement(k = "d3"), black = Placement(r = "d8", p = "d6"), whiteMove = true)
                .inCheck().shouldEqual(false)

        initBoard(white = Placement(k = "d3"), black = Placement(n = "b2"), whiteMove = true)
                .inCheck().shouldEqual(true)

        initBoard(white = Placement(k = "d3"), black = Placement(n = "c5"), whiteMove = true)
                .inCheck().shouldEqual(true)

        initBoard(white = Placement(k = "d3"), black = Placement(b = "h7"), whiteMove = true)
                .inCheck().shouldEqual(true)
    }

    @Test
    fun castleMoves() {
        // white left
        initBoard(white = Placement(ur = "a1", uk = "d1", p = "a2"),
                  black = Placement())
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a1 -> c1", "a2 -> a3", "a2 -> a4", "d1 -> b1",
                                    "d1 -> c1", "d1 -> c2", "d1 -> d2", "d1 -> e1", "d1 -> e2"))
        // white right
        initBoard(white = Placement(ur = "h1", uk = "d1", p = "h2"),
                  black = Placement())
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("d1 -> c1", "d1 -> c2", "d1 -> d2", "d1 -> e1", "d1 -> e2", "d1 -> f1",
                                    "h1 -> e1", "h1 -> f1", "h1 -> g1", "h2 -> h3", "h2 -> h4"))
        // black left
        initBoard(black = Placement(ur = "a8", uk = "d8", p = "a7"),
                  white = Placement(), whiteMove = false)
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a7 -> a5", "a7 -> a6", "a8 -> b8", "a8 -> c8", "d8 -> b8",
                                    "d8 -> c7", "d8 -> c8", "d8 -> d7", "d8 -> e7", "d8 -> e8"))
        // black right
        initBoard(black = Placement(ur = "h8", uk = "d8", p = "h7"),
                  white = Placement(), whiteMove = false)
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("d8 -> c7", "d8 -> c8", "d8 -> d7", "d8 -> e7", "d8 -> e8", "d8 -> f8",
                                    "h7 -> h5", "h7 -> h6", "h8 -> e8", "h8 -> f8", "h8 -> g8"))
        // not clear
        initBoard(white = Placement(ur = "a1", uk = "d1", p = "a2,d2", b = "c1"),
                  black = Placement())
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a2 -> a3", "a2 -> a4", "c1 -> a3", "c1 -> b2",
                                    "d1 -> c2", "d1 -> e1", "d1 -> e2", "d2 -> d3", "d2 -> d4"))
        // rook moved
        initBoard(white = Placement(r = "a1", uk = "d1", p = "a2"),
                  black = Placement())
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a1 -> c1", "a2 -> a3", "a2 -> a4",
                                    "d1 -> c1", "d1 -> c2", "d1 -> d2", "d1 -> e1", "d1 -> e2"))
        // king moved
        initBoard(white = Placement(ur = "a1", k = "d1", p = "a2"),
                  black = Placement())
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a1 -> c1", "a2 -> a3", "a2 -> a4",
                                    "d1 -> c1", "d1 -> c2", "d1 -> d2", "d1 -> e1", "d1 -> e2"))
        // from check not allowed
        initBoard(white = Placement(ur = "a1", uk = "d1", p = "a2"),
                  black = Placement(r = "d8"))
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("d1 -> c1", "d1 -> c2", "d1 -> e1", "d1 -> e2"))
        // through check not allowed
        initBoard(white = Placement(ur = "a1", uk = "d1", p = "a2"),
                  black = Placement(r = "c8"))
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a1 -> c1", "a2 -> a3", "a2 -> a4",
                                    "d1 -> d2", "d1 -> e1", "d1 -> e2"))
        // to check not allowed
        initBoard(white = Placement(ur = "a1", uk = "d1", p = "a2"),
                  black = Placement(r = "b8"))
                .nextMoves.map { it.printBoard(); it.desc() }.sorted()
                .shouldEqual(listOf("a1 -> b1", "a1 -> c1", "a2 -> a3", "a2 -> a4",
                                    "d1 -> c1", "d1 -> c2", "d1 -> d2", "d1 -> e1", "d1 -> e2"))
    }

    @Test
    fun testConversions() {
        // GameState => ModelInput => GameState (white move)
        val state1a = initBoard(white = Placement(k = "d1", q = "e5", n = "c5", r = "h7", p = "b2,c3,d2"),
                                black = Placement(k = "f7", q = "d4", p = "f6"),
                                whiteMove = true)
        val state1b = gameSpec.fromModelInput(state1a.toModelInput(), 0)
        checkStatesEqual(state1a, state1b)

        // GameState => ModelInput => GameState (black move)
        val state2a = state1a.nextMoves[0]
        val state2b = gameSpec.fromModelInput(state2a.toModelInput(), 0)
        checkStatesEqual(state2a, state2b)

        // GameState => Slim => Instance => BatchInput => GameState
        val slim = state1a.toSlimState { _, tsr -> tsr.prob = 0.5f }
        val white_win = initBoard(white = Placement(k = "a6", q = "b7"), black = Placement(k = "a8"),
                                  whiteMove = false, depth = 10)
        val white_loss = initBoard(black = Placement(k = "a6", q = "b7"), white = Placement(k = "a8"),
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
        val (input, value, policy, legal) = initTrainingData(gameSpec, 100)

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
    fun testCheckmate() {
        val final = initBoard(white = Placement(k = "d1", b = "a2,e5", r = "b7"),
                              black = Placement(k = "g8", p = "g6,h6,g5", b = "f8"), whiteMove = false)
        final.nextMoves.size.shouldEqual(0)
        final.outcome.shouldEqual(Outcome.LOSE)
    }

    @Test
    fun testStalemate() {
        val final = initBoard(white = Placement(k = "a1"),
                              black = Placement(k = "h8", q = "c2"), whiteMove = true)
        final.nextMoves.size.shouldEqual(0)
        final.outcome.shouldEqual(Outcome.DRAW)
    }

    @Test
    fun testLegalAndPolicy() {
        val state1 = initBoard(white = Placement(k = "d1", b = "a2,e1", r = "b1"),
                               black = Placement(k = "h8", p = "g7,h7,g5", b = "h6"), whiteMove = false)
        val final = initBoard(white = Placement(k = "d1", b = "a2,e5", r = "b7"),
                              black = Placement(k = "g8", p = "g6,h6,g5", b = "f8"), whiteMove = false)
        val params = SearchParameters(exploration = 0.2, temperature = 0.01, iterations = 5)
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

        // Two legal moves:
        val g5g4 = gameSpec.toPolicyIndex(MoveInfo(6, 4, 6, 3, 0))
        val g7g6 = gameSpec.toPolicyIndex(MoveInfo(6, 6, 6, 5, 0))
        checkLegalAndPolicy(0, intArrayOf(g5g4, g7g6))

        val b5b4 = gameSpec.toPolicyIndex(MoveInfo(1, 4, 1, 3, 0))
        val b7b6 = gameSpec.toPolicyIndex(MoveInfo(1, 6, 1, 5, 0))
        checkLegalAndPolicy(1, intArrayOf(b5b4, b7b6))

        val g2g3 = gameSpec.toPolicyIndex(MoveInfo(6, 1, 6, 2, 0))
        val g4g5 = gameSpec.toPolicyIndex(MoveInfo(6, 3, 6, 4, 0))
        checkLegalAndPolicy(2, intArrayOf(g2g3, g4g5))

        val b2b3 = gameSpec.toPolicyIndex(MoveInfo(1, 1, 1, 2, 0))
        val b4b5 = gameSpec.toPolicyIndex(MoveInfo(1, 3, 1, 4, 0))
        checkLegalAndPolicy(3, intArrayOf(b2b3, b4b5))
    }

    @Test
    fun testThreefoldRepetition() {
        var state: GameState = ChessState(gameSpec)
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
}
