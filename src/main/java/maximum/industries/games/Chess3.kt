package maximum.industries.games

import maximum.industries.*
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

    // Returns true if the king is attacked from a given direction within a given distance
    // by a piece other than a knight. This is called *alot* needs to be efficient.
    fun attacked(board: ByteArray, king: Int, offset: Int, maxMove: Int, otp: Int): Boolean {
        for (i in 1..maxMove) {
            val piece = board[king + i * offset] * otp
            when (piece) {
                NONE -> {
                    // No piece in this square, continue looking in this direction
                }
                PAWN -> {
                    // Pawn only attacks forwards
                    return i == 1 && offset.sign == otp.sign
                }
                KNIGHT -> {
                    return false
                }
                BISHOP -> {
                    return (offset == -9 || offset == -7 || offset == 7 || offset == 9)
                }
                ROOK, UROOK -> {
                    return (offset == -8 || offset == 1 || offset == 8 || offset == -1)
                }
                QUEEN -> {
                    return true
                }
                KING, UKING -> {
                    return i == 1
                }
                else -> {
                    // The else condition matches negatives, which are own pieces.
                    return false
                }
            }
        }
        return false
    }

    fun inCheck(board: ByteArray, kx: Int, ky: Int): Boolean {
        if (kx < 0 && ky < 0) return false // should happen in tests only
        val kpos = ky * 8 + kx
        val otp = if (board[kpos] > 0) -1 else 1 // "opponent to positive"
        val rx = 7 - kx  // remaining x past kx
        val ry = 7 - ky  // remaining y past ky

        if (attacked(board, kpos, -8, ky, otp)) return true
        if (attacked(board, kpos, -7, if (ky < rx) ky else rx, otp)) return true
        if (attacked(board, kpos, +1, rx, otp)) return true
        if (attacked(board, kpos, +9, if(ry < rx) ry else rx, otp)) return true
        if (attacked(board, kpos, +8, ry, otp)) return true
        if (attacked(board, kpos, +7, if (ry < kx) ry else kx, otp)) return true
        if (attacked(board, kpos, -1, kx, otp)) return true
        if (attacked(board, kpos, -9, if (kx < ky) kx else ky, otp)) return true

        val oppKnight = (otp * KNIGHT) as Byte
        if (kx > 0) {
            if (ky > 1 && board[kpos - 17] == oppKnight) return true
            if (ky < 6 && board[kpos + 15] == oppKnight) return true
            if (kx > 1 && ky > 0 && board[kpos - 10] == oppKnight) return true
            if (kx > 1 && ky < 7 && board[kpos + 6] == oppKnight) return true
        }
        if (kx < 7) {
            if (ky > 1 && board[kpos - 15] == oppKnight) return true
            if (ky < 6 && board[kpos + 17] == oppKnight) return true
            if (kx < 6 && ky > 0 && board[kpos - 6] == oppKnight) return true
            if (kx < 6 && ky < 7 && board[kpos + 10] == oppKnight) return true
        }
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
    // returns true if x2,y2 was an empty square (so linear move search should continue)
    fun maybeMove(states: ArrayList<GameState>,
                  x1: Int, y1: Int,
                  x2: Int, y2: Int,
                  kx: Int, ky: Int,
                  p1: Int = at(x1, y1)): Boolean {
        if (x2 < 0 || x2 > 7 || y2 < 0 || y2 > 7) return false // off board

        val p2 = at(x2, y2)
        if (p1.sign == p2.sign) return false // can't capture or go past own piece

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

    fun initNextMoves(stopAtOne: Boolean): ArrayList<GameState> {
        val states = ArrayList<GameState>()
        val playerSign = if (player.eq(Player.WHITE)) 1 else -1
        val (kx, ky) = kingPos()

        val board = gameBoard
        for (i in 0 until BOARD_SQUARES) {
            val x = i % BOARD_SIZE
            val y = i / BOARD_SIZE
            val p = board[i].toInt()

            when (p * playerSign) {
                // TODO: pawn efficiency, en passant
                PAWN -> {
                    val p1 = if (y + playerSign == 0 || y + playerSign == 7) {
                        QUEEN * playerSign
                    } else {
                        PAWN * playerSign
                    }
                    if (at(x, y + playerSign) == 0) {
                        // push 1
                        maybeMove(states, x, y, x, y + playerSign, kx, ky, p1)
                        // initial push 2
                        if ((y - playerSign) % 7 == 0 && at(x, y + 2 * playerSign) == 0) {
                            maybeMove(states, x, y, x, y + 2 * playerSign, kx, ky)
                        }
                    }
                    // capture left
                    if (x > 0 && at(x - 1, y + playerSign) != 0) {
                        maybeMove(states, x, y, x - 1, y + playerSign, kx, ky, p1)
                    }
                    // capture right
                    if (x < 7 && at(x + 1, y + playerSign) != 0) {
                        maybeMove(states, x, y, x + 1, y + playerSign, kx, ky, p1)
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
                    if (p * playerSign == UKING) {
                        maybeCastle(states, x, y)
                    }
                }
                else -> {
                    // empty square or opponent piece
                }
            }
            if (stopAtOne && states.size > 0) break
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
            if (count >= 3) return Outcome.DRAW
        }
        return Outcome.UNDETERMINED
    }
}

