package max.dillon

import com.google.protobuf.TextFormat
import junit.framework.Assert.assertEquals
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

fun initBoard(white: Placement, black: Placement, whiteMove: Boolean): GameState {
    val gameSpec = getGameSpec()
    val state = GameState(gameSpec)
    state.whiteMove = whiteMove
    state.gameBoard = Array(gameSpec.boardSize, { IntArray(gameSpec.boardSize, { 0 }) })

    fun place(placement: Placement, sign: Int) {
        val (p, n, b, r, q, k) = placement
        arrayListOf<String>(p, n, b, r, q, k).forEachIndexed { i, pstr ->
            if (pstr.length > 0) {
                for (p in pstr.split(",")) {
                    val x = p.first() - 'a'
                    val y = p.substring(1).toInt() - 1
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
        assertEquals(2, initBoard(
                white = Placement(p = "a2"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(1, initBoard(
                white = Placement(p = "a3"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(16, initBoard(
                white = Placement(p = "a2,b2,c2,d2,e2,f2,g2,h2"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(2, initBoard(
                white = Placement(p = "a7"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(0, initBoard(
                white = Placement(p = "a3"),
                black = Placement(p = "a4"),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(0, initBoard(
                white = Placement(p = "a3,a4"),
                black = Placement(p = "a5"),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(3, initBoard(
                white = Placement(p = "b3"),
                black = Placement(p = "a4,c4"),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(1, initBoard(
                white = Placement(p = "a4,b3,c4"),
                black = Placement(p = "a5,c5"),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(4, initBoard(
                white = Placement(p = "b2"),
                black = Placement(p = "a3,c3"),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(4, initBoard(
                white = Placement(p = "a6,c6"),
                black = Placement(p = "b7"),
                whiteMove = false)
                .getLegalNextStates().size)

        assertEquals(2, initBoard(
                white = Placement(),
                black = Placement(p = "b2"),
                whiteMove = false)
                .getLegalNextStates().size)
    }

    @Test
    fun knightMoves() {
        assertEquals(3, initBoard(
                white = Placement(n = "b1"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(10, initBoard(
                white = Placement(p = "a2,b2,c2,d2", n = "b1"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)

        assertEquals(8, initBoard(
                white = Placement(n = "d4"),
                black = Placement(),
                whiteMove = true)
                .getLegalNextStates().size)
    }
}

