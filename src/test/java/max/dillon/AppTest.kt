package max.dillon

import com.google.protobuf.TextFormat
import org.amshove.kluent.shouldEqual
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Paths

fun getGameSpec(): GameGrammar.GameSpec {
    val specStr = String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")))
    val gameSpec = GameGrammar.GameSpec.newBuilder().apply {
        TextFormat.getParser().merge(specStr, this)
        addPiece(0, addPieceBuilder()) // hack: inserting null piece at index 0
    }.build()
    return gameSpec
}

data class Placement(
        val p: String = "", val n: String = "", val b: String = "",
        val r: String = "", val q: String = "", val k: String = "")

fun initBoard(white: Placement, black: Placement, whiteMove: Boolean = true): GameState {
    val gameSpec = getGameSpec()
    val state = GameState(gameSpec)
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

class TestApp() {
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
}

