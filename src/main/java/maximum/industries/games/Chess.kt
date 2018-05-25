package maximum.industries.games

import maximum.industries.*
import java.util.*
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sign

const val PAWN = 1
const val KNIGHT = 2
const val BISHOP = 3
const val ROOK = 4
const val QUEEN = 5
const val KING = 6
const val UROOK = 7
const val UKING = 8

class ChessState : GameState {

    constructor(gameSpec: GameGrammar.GameSpec) : super(gameSpec) {}

    constructor(gameSpec: GameGrammar.GameSpec,
                gameBoard: ByteArray, player: Player,
                p1: Int,
                x1: Int, y1: Int,
                x2: Int, y2: Int,
                moveDepth: Short,
                history: IntArray = IntArray(16) { 0 }) :
            super(gameSpec, gameBoard, player, p1, x1, y1, x2, y2, moveDepth, history) {
    }

    fun attacked(board: ByteArray, king: Int, offset: Int, maxMove: Int): Boolean {
        val otp = if (board[king] > 0) -1 else 1 // "opponent to positive"
        val diagonal = (abs(offset) and (abs(offset) - 1)) > 0 // diag if 2+ bits set
        for (i in 1..maxMove) {
            val piece = board[king + i * offset] * otp
            if (piece != 0) {
                if (piece < 0) {
                    return false // first piece is own piece
                } else if (diagonal) {
                    if (piece == QUEEN || piece == BISHOP) return true
                    else if (i == 1) {
                        if (piece == KING || piece == UKING) return true
                        if (offset.sign != otp.sign && piece == PAWN) return true
                    } else return false
                } else {
                    if (piece == QUEEN || piece == ROOK || piece == UROOK) return true
                    else if (i == 1 && (piece == KING || piece == UKING)) return true
                    else return false
                }
            }
        }
        return false
    }

    fun inCheck(board: ByteArray, kx: Int, ky: Int): Boolean {
        if (kx < 0 && ky < 0) return false // should happen in tests only

        val kpos = ky * 8 + kx
        if (attacked(board, kpos, -8, ky)) return true
        if (attacked(board, kpos, -7, min(ky, 7 - kx))) return true
        if (attacked(board, kpos, +1, 7 - kx)) return true
        if (attacked(board, kpos, +9, min(7 - ky, 7 - kx))) return true
        if (attacked(board, kpos, +8, 7 - ky)) return true
        if (attacked(board, kpos, +7, min(7 - ky, kx))) return true
        if (attacked(board, kpos, -1, kx)) return true
        if (attacked(board, kpos, -9, min(kx, ky))) return true

        val otp = if (board[kpos] > 0) -1 else 1 // "opponent to positive"
        if (kx > 0 && ky > 1 && board[kpos - 17] * otp == KNIGHT) return true
        if (kx < 7 && ky > 1 && board[kpos - 15] * otp == KNIGHT) return true
        if (kx < 6 && ky > 0 && board[kpos - 6] * otp == KNIGHT) return true
        if (kx < 6 && ky < 7 && board[kpos + 10] * otp == KNIGHT) return true
        if (kx < 7 && ky < 6 && board[kpos + 17] * otp == KNIGHT) return true
        if (kx > 0 && ky < 6 && board[kpos + 15] * otp == KNIGHT) return true
        if (kx > 1 && ky < 7 && board[kpos + 6] * otp == KNIGHT) return true
        if (kx > 1 && ky > 0 && board[kpos - 10] * otp == KNIGHT) return true

        return false
    }

    fun inCheck(): Boolean {
        val (kx, ky) = kingPos()
        return inCheck(gameBoard, kx, ky)
    }

    fun set(board: ByteArray, x: Int, y: Int, p: Int) {
        board[y * gameSpec.boardSize + x] = p.toByte()
    }

    // adds a move if the move is legal
    // returns true if x2,y2 was an empty square
    fun maybeMove(states: ArrayList<GameState>,
                  x1: Int, y1: Int,
                  x2: Int, y2: Int,
                  kx: Int, ky: Int,
                  p1: Int = at(x1, y1)): Boolean {
        if (x2 < 0 || x2 > 7 || y2 < 0 || y2 > 7) return false // off board

        val p2 = at(x2, y2)
        if (p1.sign == p2.sign) return false // can't capture own piece

        val nextBoard = gameBoard.clone()
        set(nextBoard, x2, y2, p1)
        set(nextBoard, x1, y1, 0)

        val kkx = if (kx == -1) x2 else kx
        val kky = if (ky == -1) y2 else ky

        if (!inCheck(nextBoard, kkx, kky)) {
            val nextPlayer = if (p1.sign < 0) Player.WHITE else Player.BLACK
            states.add(ChessState(gameSpec, nextBoard, nextPlayer, p1, x1, y1, x2, y2,
                                  (moveDepth + 1).toShort(), history.clone()))
        }
        return p2 == 0 // return true only if destination was empty
    }

    fun maybeCastle(states: ArrayList<GameState>, xk: Int, yk: Int) {
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

    fun addLinear(states: ArrayList<GameState>,
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

    fun kingPos(): Pair<Int, Int> {
        for (i in 0 until gameBoard.size) {
            val type = gameBoard[i].toInt() * if (player.eq(Player.WHITE)) 1 else -1
            if (type == KING || type == UKING) {
                return Pair(i % gameSpec.boardSize, i / gameSpec.boardSize)
            }
        }
        return Pair(-2, -2)
    }

    override fun initNextMoves(): ArrayList<GameState> {
        return initNextMoves(false)
    }

    fun initNextMoves(stopAtOne: Boolean): ArrayList<GameState> {
        val states = ArrayList<GameState>()
        val playerSign = if (player.eq(Player.WHITE)) 1 else -1
        val (kx, ky) = kingPos()

        for (i in 0 until gameBoard.size) {
            val x = i % gameSpec.boardSize
            val y = i / gameSpec.boardSize
            val p = gameBoard[i].toInt()

            fun addPawnMoves() {
                val p1 = if (y + playerSign == 0 || y + playerSign == 7) {
                    QUEEN * playerSign
                } else {
                    PAWN * playerSign
                }
                if (at(x, y + playerSign) == 0) {
                    if ((y - playerSign) % 7 == 0 && at(x, y + 2 * playerSign) == 0) {
                        maybeMove(states, x, y, x, y + 2 * playerSign, kx, ky)
                    }
                    maybeMove(states, x, y, x, y + playerSign, kx, ky, p1)
                }
                if (x > 0 && at(x - 1, y + playerSign) != 0) {
                    maybeMove(states, x, y, x - 1, y + playerSign, kx, ky, p1)
                }
                if (x < 7 && at(x + 1, y + playerSign) != 0) {
                    maybeMove(states, x, y, x + 1, y + playerSign, kx, ky, p1)
                }
            }

            fun addKnightMoves() {
                maybeMove(states, x, y, x - 1, y - 2, kx, ky)
                maybeMove(states, x, y, x + 1, y - 2, kx, ky)
                maybeMove(states, x, y, x - 1, y + 2, kx, ky)
                maybeMove(states, x, y, x + 1, y + 2, kx, ky)
                maybeMove(states, x, y, x - 2, y - 1, kx, ky)
                maybeMove(states, x, y, x + 2, y - 1, kx, ky)
                maybeMove(states, x, y, x - 2, y + 1, kx, ky)
                maybeMove(states, x, y, x + 2, y + 1, kx, ky)
            }

            fun addBishopMoves() {
                addLinear(states, x, y, kx, ky, false, true, 7, p, stopAtOne)
            }

            fun addRookMoves() {
                addLinear(states, x, y, kx, ky, true, false, 7, ROOK * p.sign, stopAtOne)
            }

            fun addQueenMoves() {
                addLinear(states, x, y, kx, ky, true, true, 7, p, stopAtOne)
            }

            fun addKingMoves() {
                addLinear(states, x, y, -1, -1, true, true, 1, KING * p.sign, stopAtOne)
                if (abs(at(x, y)) == UKING) {
                    maybeCastle(states, x, y)
                }
            }

            when (p * playerSign) {
                PAWN -> addPawnMoves()
                KNIGHT -> addKnightMoves()
                BISHOP -> addBishopMoves()
                ROOK -> addRookMoves()
                QUEEN -> addQueenMoves()
                KING -> addKingMoves()
                UROOK -> addRookMoves()
                UKING -> addKingMoves()
                else -> {
                    // empty square or opponent piece
                }
            }
            if (states.size > 0 && stopAtOne) break
        }
        return states
    }

    override fun gameOutcome(): Outcome {
        if (initNextMoves(true).size == 0) {
            if (inCheck()) {
                return Outcome.LOSE
            } else {
                return Outcome.DRAW
            }
        } else if (moveDepth >= 200) {
            return Outcome.DRAW
        } else {
            var count = 0
            var lastHash = history[moveDepth % 16]
            for (hash in history) {
                if (hash == lastHash) count++
            }
            if (count == 3) return Outcome.DRAW
        }
        return Outcome.UNDETERMINED
    }
}

