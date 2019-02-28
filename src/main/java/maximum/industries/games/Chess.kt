package maximum.industries.games

import maximum.industries.*
import java.lang.RuntimeException
import java.util.*
import kotlin.math.abs
import kotlin.math.sign

const val BOARD_SIZE = 8
const val BOARD_SQUARES = 64

const val NONE = 0
const val PAWN = 1
const val KNIGHT = 2
const val BISHOP = 3
const val ROOK = 4
const val QUEEN = 5
const val KING = 6
const val UROOK = 7
const val UKING = 8

class ChessState : GameState {

    constructor(gameSpec: GameGrammar.GameSpec) : super(gameSpec)

    constructor(gameSpec: GameGrammar.GameSpec,
                gameBoard: ByteArray, player: Player,
                p1: Int,
                x1: Int, y1: Int,
                x2: Int, y2: Int,
                moveDepth: Short,
                history: IntArray = IntArray(16) { 0 }) :
            super(gameSpec, gameBoard, player, p1, x1, y1, x2, y2, moveDepth, history)

    companion object {
        private val charToPiece = mapOf(
                'P' to 1, 'N' to 2, 'B' to 3, 'R' to 4, 'Q' to 5, 'K' to 6,
                'p' to -1, 'n' to -2, 'b' to- 3, 'r' to -4, 'q' to -5, 'k' to -6)
        private val pieceToChar = charToPiece.entries.associate { (k, v) -> v to k }
                .toMutableMap().also {
                    it[-7] = 'r'
                    it[-8] = 'k'
                    it[7] ='R'
                    it[8] = 'K' }

        fun fromFen(gameSpec: GameGrammar.GameSpec, fen: String): ChessState {
            val fenToks = fen.split(' ')
            val player =  if (fenToks.contains("w")) Player.WHITE else Player.BLACK
            val depth = fenToks.last().toInt() * 2 - if (player == Player.WHITE) 2 else 1
            val gameBoard = ByteArray(64)
            for ((row, rowStr) in fenToks[0].split("/").reversed().withIndex()) {
                var col = 0
                for (c in rowStr) {
                    if (c.isDigit()) {
                        col += c - '0'
                    } else {
                        gameBoard[row * 8 + col] = charToPiece[c]!!.toByte()
                        col++
                    }
                }
            }
            val castling = fenToks[2]
            if ('K' in castling) {
                gameBoard[0 * 8 + 4] = 8
                gameBoard[0 * 8 + 7] = 7
            }
            if ('Q' in castling) {
                gameBoard[0 * 8 + 4] = 8
                gameBoard[0 * 8 + 0] = 7
            }
            if ('k' in castling) {
                gameBoard[7 * 8 + 4] = -8
                gameBoard[7 * 8 + 7] = -7
            }
            if ('q' in castling) {
                gameBoard[7 * 8 + 4] = -8
                gameBoard[7 * 8 + 0] = -7
            }
            return ChessState(gameSpec, gameBoard, player, 0, 0, 0, 0, 0, depth.toShort())
        }
    }

    fun fen(): String {
        val rows = List(8) {
            var line = ""
            var skips = 0
            for (col in 0 until 8) {
                if (at(col, 7 - it) == 0) {
                    skips++
                } else {
                    if (skips > 0) {
                        line += skips.toString()
                    }
                    line += pieceToChar[at(col, 7 - it)]
                    skips = 0
                }
            }
            if (skips > 0) {
                line += skips.toString()
            }
            line
        }
        val move = if (player == Player.WHITE) "w" else "b"
        var castling = ""
        if (at(7,0) == 7 && at(4, 0) == 8) castling += "K"
        if (at(0,0) == 7 && at(4, 0) == 8) castling += "Q"
        if (at(7,7) == -7 && at(4, 7) == -8) castling += "k"
        if (at(0,7) == -7 && at(4, 7) == -8) castling += "q"
        if (castling == "") castling = "-"
        val fullMove = Math.max(1, (moveDepth + 1) / 2 + if(player == Player.WHITE) 1 else 0)
        return "${rows.joinToString("/")} $move $castling - - $fullMove"
    }

    fun uci(): String {
        val promoted = Math.abs(p1) == 1 && get(x2, y2) != p1
        return "${'a' + x1}${y1 + 1}${'a' + x2}${y2 + 1}${if (promoted) "q" else ""}"
    }

    fun move(uci: String): ChessState {
        for (next in nextMoves) {
            val state = next as ChessState
            if (state.uci() == uci) {
                return state
            }
        }
        throw RuntimeException("move not valid")
    }

    // this is called *alot* needs to be efficient.
    private fun attacked(board: ByteArray, king: Int, offset: Int, maxMove: Int): Boolean {
        val otp = if (board[king] > 0) -1 else 1 // "opponent to positive"
        //val diagonal = (abs(offset) and (abs(offset) - 1)) > 0 // diag if 2+ bits set
        val diagonal = (offset == -9 || offset == -7 || offset == 7 || offset == 9) // faster
        for (i in 1..maxMove) {
            val piece = board[king + i * offset] * otp
            if (piece != 0) {
                return if (piece < 0) {
                    false // first piece is own piece
                } else if (diagonal) {
                    if (piece == QUEEN || piece == BISHOP) true
                    else if (i == 1) {
                        if (piece == KING || piece == UKING) true
                        else offset.sign != otp.sign && piece == PAWN
                    } else false
                } else {
                    if (piece == QUEEN || piece == ROOK || piece == UROOK) true
                    else i == 1 && (piece == KING || piece == UKING)
                }
            }
        }
        return false
    }

    private fun inCheck(board: ByteArray, kx: Int, ky: Int): Boolean {
        if (kx < 0 && ky < 0) return false // should happen in tests only

        val kpos = ky * 8 + kx
        val smkx = 7 - kx // we'll use these in inlining min() calls to avoid null checks
        val smky = 7 - ky
        if (attacked(board, kpos, -8, ky)) return true
        if (attacked(board, kpos, -7, if (ky < smkx) ky else smkx)) return true
        if (attacked(board, kpos, +1, smkx)) return true
        if (attacked(board, kpos, +9, if(smky < smkx) smky else smkx)) return true
        if (attacked(board, kpos, +8, smky)) return true
        if (attacked(board, kpos, +7, if (smky < kx) smky else kx)) return true
        if (attacked(board, kpos, -1, kx)) return true
        if (attacked(board, kpos, -9, if (kx < ky) kx else ky)) return true

        val otp = if (board[kpos] > 0) -1 else 1 // "opponent to positive"
        if (kx > 0 && ky > 1 && board[kpos - 17] * otp == KNIGHT) return true
        if (kx < 7 && ky > 1 && board[kpos - 15] * otp == KNIGHT) return true
        if (kx > 1 && ky > 0 && board[kpos - 10] * otp == KNIGHT) return true
        if (kx < 6 && ky > 0 && board[kpos - 6] * otp == KNIGHT) return true
        if (kx > 1 && ky < 7 && board[kpos + 6] * otp == KNIGHT) return true
        if (kx < 6 && ky < 7 && board[kpos + 10] * otp == KNIGHT) return true
        if (kx > 0 && ky < 6 && board[kpos + 15] * otp == KNIGHT) return true
        if (kx < 7 && ky < 6 && board[kpos + 17] * otp == KNIGHT) return true

        return false
    }

    fun inCheck(): Boolean {
        val (kx, ky) = kingPos()
        return inCheck(gameBoard, kx, ky)
    }

    fun set(board: ByteArray, x: Int, y: Int, p: Int) {
        board[y * BOARD_SIZE + x] = p.toByte()
    }

    // adds a move if the move is legal
    // returns true if x2,y2 was an empty square
    private fun maybeMove(states: ArrayList<GameState>,
                  x1: Int, y1: Int,
                  x2: Int, y2: Int,
                  kx: Int, ky: Int,
                  p1: Int = at(x1, y1)): Boolean {
        if (x2 < 0 || x2 > 7 || y2 < 0 || y2 > 7) return false // off board

        val porig = at(x1, y1)
        val p2 = at(x2, y2)
        if (p1.sign == p2.sign) return false // can't capture own piece

        val nextBoard = gameBoard.clone()
        set(nextBoard, x2, y2, p1)
        set(nextBoard, x1, y1, 0)

        // en passant
        if (abs(porig) == PAWN && x1 != x2 && p2 == 0) {
            set(nextBoard, x2, y1, 0)
        }

        // tidy up castling rights
        if (y1 == 0 || y1 == 7) {
            if (abs(porig) == UKING) {
                if (abs(at(0, y1)) == UROOK) set(nextBoard, 0, y1, ROOK * porig.sign)
                if (abs(at(7, y1)) == UROOK) set(nextBoard, 7, y1, ROOK * porig.sign)
            }
            if (abs(porig) == UROOK) {
                if (abs(at(7 - x1, y1)) != UROOK && abs(at(4, y1)) == UKING) {
                    set(nextBoard, 4, y1, KING * porig.sign)
                }
            }
        }

        val kkx = if (kx == -1) x2 else kx
        val kky = if (ky == -1) y2 else ky

        if (!inCheck(nextBoard, kkx, kky)) {
            val nextPlayer = if (p1.sign < 0) Player.WHITE else Player.BLACK
            states.add(ChessState(gameSpec, nextBoard, nextPlayer, porig, x1, y1, x2, y2,
                                  (moveDepth + 1).toShort(), history.clone()))
        }
        return p2 == 0 // return true only if destination was empty
    }

    private fun maybeCastle(states: ArrayList<GameState>, xk: Int, yk: Int) {
        val sign = at(xk, yk).sign
        if (abs(at(0, yk)) == UROOK) {
            var clear = true
            for (x in 1 until xk) if (at(x, yk) != 0) clear = false
            if (clear && !inCheck()) {
                val nextBoard = gameBoard.clone()
                set(nextBoard, xk, yk, 0)
                set(nextBoard, xk - 1, yk, KING * sign)
                if (!inCheck(nextBoard, xk - 1, yk)) {
                    set(nextBoard, xk - 1, yk, ROOK * sign)
                    set(nextBoard, xk - 2, yk, KING * sign)
                    set(nextBoard, 0, yk, 0)
                    if (abs(at(7, yk)) == UROOK) set(nextBoard, 7, yk, ROOK * sign)
                    if (!inCheck(nextBoard, xk - 2, yk)) {
                        val nextPlayer = if (sign < 0) Player.WHITE else Player.BLACK
                        states.add(ChessState(gameSpec, nextBoard, nextPlayer,
                                              KING, xk, yk, xk - 2, yk,
                                              (moveDepth + 1).toShort(), history.clone()))
                    }
                }
            }
        }
        if (abs(at(7, yk)) == UROOK) {
            var clear = true
            for (x in (xk + 1) until 7) if (at(x, yk) != 0) clear = false
            if (clear && !inCheck()) {
                val nextBoard = gameBoard.clone()
                set(nextBoard, xk, yk, 0)
                set(nextBoard, xk + 1, yk, KING * sign)
                if (!inCheck(nextBoard, xk + 1, yk)) {
                    set(nextBoard, xk + 1, yk, ROOK * sign)
                    set(nextBoard, xk + 2, yk, KING * sign)
                    set(nextBoard, 7, yk, 0)
                    if (abs(at(0, yk)) == UROOK) set(nextBoard, 0, yk, ROOK * sign)
                    if (!inCheck(nextBoard, xk + 2, yk)) {
                        val nextPlayer = if (sign < 0) Player.WHITE else Player.BLACK
                        states.add(ChessState(gameSpec, nextBoard, nextPlayer,
                                              KING, xk, yk, xk + 2, yk,
                                              (moveDepth + 1).toShort(), history.clone()))
                    }
                }
            }
        }
    }

    private fun addLinear(states: ArrayList<GameState>,
                  x1: Int, y1: Int, kx: Int, ky: Int,
                  rook: Boolean, bishop: Boolean,
                  dist: Int, piece: Int, stopAtOne: Boolean) {
        if (rook) {
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 + i, y1 + 0, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 + 0, y1 + i, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 - i, y1 + 0, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 + 0, y1 - i, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
        }
        if (bishop) {
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 + i, y1 + i, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 + i, y1 - i, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 - i, y1 + i, kx, ky, piece)) break
            if (stopAtOne && states.size > 0) return
            for (i in 1..dist) if (!maybeMove(states, x1, y1, x1 - i, y1 - i, kx, ky, piece)) break
        }
    }

    private fun kingPos(): Pair<Int, Int> {
        val sign = if (player.eq(Player.WHITE)) 1 else -1
        val board = gameBoard
        for (i in 0 until BOARD_SQUARES) {
            val type = board[i].toInt() * sign
            if (type == KING || type == UKING) {
                return Pair(i % BOARD_SIZE, i / BOARD_SIZE)
            }
        }
        return Pair(-2, -2)
    }

    override fun initNextMoves(): ArrayList<GameState> {
        return initNextMoves(false)
    }

    private fun initNextMoves(stopAtOne: Boolean): ArrayList<GameState> {
        val states = ArrayList<GameState>()
        val playerSign = if (player.eq(Player.WHITE)) 1 else -1
        val (kx, ky) = kingPos()

        val board = gameBoard
        for (i in 0 until BOARD_SQUARES) {
            val x = i % BOARD_SIZE
            val y = i / BOARD_SIZE
            val p = board[i].toInt()

            when (p * playerSign) {
                PAWN -> {
                    val p1 = if (y + playerSign == 0 || y + playerSign == 7) {
                        QUEEN * playerSign
                    } else {
                        PAWN * playerSign
                    }
                    // push one or two
                    if (at(x, y + playerSign) == 0) {
                        if ((y - playerSign) % 7 == 0 && at(x, y + 2 * playerSign) == 0) {
                            maybeMove(states, x, y, x, y + 2 * playerSign, kx, ky)
                        }
                        maybeMove(states, x, y, x, y + playerSign, kx, ky, p1)
                    }
                    // captures
                    if (x > 0 && at(x - 1, y + playerSign) != 0) {
                        maybeMove(states, x, y, x - 1, y + playerSign, kx, ky, p1)
                    }
                    if (x < 7 && at(x + 1, y + playerSign) != 0) {
                        maybeMove(states, x, y, x + 1, y + playerSign, kx, ky, p1)
                    }
                    // en passant
                    if (((y1 == 1 && y2 == 3) || (y1 == 6 && y2 == 4)) && abs(at(x2, y2)) == PAWN) {
                        if (y == y2 && abs(x - x2) == 1) {
                            maybeMove(states, x, y, x2, y + playerSign, kx, ky, p1)
                        }
                    }
                }
                KNIGHT -> {
                    maybeMove(states, x, y, x - 1, y - 2, kx, ky)
                    maybeMove(states, x, y, x + 1, y - 2, kx, ky)
                    maybeMove(states, x, y, x - 1, y + 2, kx, ky)
                    maybeMove(states, x, y, x + 1, y + 2, kx, ky)
                    maybeMove(states, x, y, x - 2, y - 1, kx, ky)
                    maybeMove(states, x, y, x + 2, y - 1, kx, ky)
                    maybeMove(states, x, y, x - 2, y + 1, kx, ky)
                    maybeMove(states, x, y, x + 2, y + 1, kx, ky)
                }
                BISHOP -> {
                    addLinear(states, x, y, kx, ky, false, true, 7, p, stopAtOne)
                }
                ROOK, UROOK -> {
                    addLinear(states, x, y, kx, ky, true, false, 7, ROOK * p.sign, stopAtOne)
                }
                QUEEN -> {
                    addLinear(states, x, y, kx, ky, true, true, 7, p, stopAtOne)
                }
                KING, UKING -> {
                    addLinear(states, x, y, -1, -1, true, true, 1, KING * p.sign, stopAtOne)
                    if (abs(at(x, y)) == UKING) {
                        maybeCastle(states, x, y)
                    }
                }
                else -> {
                    // empty square or opponent piece
                }
            }
            if (stopAtOne && states.size > 0) break
        }
        // uncomment for determinism across game impls
        //states.sortBy { "${'a' + it.x1}${it.y1 + 1}${'a' + it.x2}${it.y2 + 1}" }
        return states
    }

    override fun gameOutcome(): Outcome {
        // no legal moves
        if (initNextMoves(true).size == 0) {
            return if (inCheck()) {
                Outcome.LOSE  // checkmate
            } else {
                Outcome.DRAW  // stalemate
            }
        }
        // insufficient material (kk, kkb, kkn)
        var nb = 0
        var prq = 0
        for (p in gameBoard) {
            val pa = abs(p.toInt())
            when (pa) {
                KNIGHT, BISHOP -> nb++
                PAWN, ROOK, UROOK, QUEEN -> prq++
            }
        }
        if (prq == 0 && nb <= 1) {
            return Outcome.DRAW
        }
        // game length bound
        if (moveDepth >= 200) {
            return Outcome.DRAW
        }
        // threefold repetition
        var count = 0
        val lastHash = history[moveDepth % 16]
        for (hash in history) {
            if (hash == lastHash) count++
        }
        if (count == 3) return Outcome.DRAW

        return Outcome.UNDETERMINED
    }
}

