package maximum.industries

import io.ktor.application.*
import io.ktor.content.default
import io.ktor.content.files
import io.ktor.content.static
import io.ktor.features.ContentNegotiation
import io.ktor.request.receive
import io.ktor.response.*
import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.gson.*
import org.amshove.kluent.shouldBeInstanceOf


/*
Example:
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"board": [1,2,3,4,0,0,0,0], "moveDepth": 3, "whiteMove": false, "moves": [{"first":{"first":1, "second":2}, "second":{"first":2, "second":3}}] }' \
     http://localhost:8080/
*/
fun main(args: Array<String>) {
    val gameSpec = loadSpec("chess")
    var state = GameState(gameSpec)

    val server = embeddedServer(Netty, port = 8080) {
        var white = getAlgo("mcts",SearchParameters(1,1.0,1.0,1))
        var black = getAlgo("mcts",SearchParameters(1,1.0,1.0,1))
        var playerTurnWhite = true

        install(ContentNegotiation) {
            gson {
                setPrettyPrinting()
            }
        }
        routing {


            post("/start") {
                playerTurnWhite = call.receive<Boolean>()
                state = GameState(gameSpec)
                val whiteAlgo = if (playerTurnWhite) "gui" else "mcts"
                white = getAlgo(whiteAlgo)

                val blackAlgo = if (playerTurnWhite) "mcts" else "gui".also { state = white.next(state).first}
                black = getAlgo(blackAlgo)

                call.respond(Pair(state.toWireState(),gameSpec))
            }


            post("/move") {
                val received = call.receive<WireState>()
                val sig = received.state.hashCode()
                if (true) {
                    if (playerTurnWhite) {
                        white.index(received.moveIndex)
                        state = white.next(received.toGameState(gameSpec)).first
                        state = black.next(state).first
                    }
                    else {
                        black.index(received.moveIndex)
                        state = black.next(received.toGameState(gameSpec)).first
                        state = white.next(state).first
                    }

                    call.respond(state.toWireState())

                } else {
                    call.respond(received)
                }

            }

            static ("/") {
                default("static/webpage.html")
                files("static")
            }

        }
    }
    server.start(wait = true)
}

data class ObjectState (val board: ByteArray,
                  val moves: IntArray,
                  val whiteMove: Boolean,
                  val moveDepth: Int) {
    override fun hashCode(): Int {
        val modulo = 7529350
        val secretKey = 58327403923465493
        val boardHash = board.contentToString().hashCode()
        return ((secretKey*boardHash*whiteMove.hashCode() * moveDepth.hashCode())%modulo).toInt()
    }
    override fun equals(other: Any?): Boolean {
        if (other !is ObjectState){
            return false
        } else return board.contentEquals(other.board) &&
                moves.contentEquals(other.moves) &&
                whiteMove == other.whiteMove &&
                moveDepth == other.moveDepth

    }
}

data class WireState(val state: ObjectState,
                     val moveIndex: Int,
                     val signature: Int) {

    override fun hashCode(): Int {
        val modulo = 78621349
        val secretKey = 58327403923465493 % modulo
        return ((secretKey * state.hashCode())%modulo).toInt()
    }
    override fun equals(other: Any?): Boolean {
        if (other !is WireState) {
            return false
        } else return state.equals(other.state)
    }

}



class GuiInput : GameSearchAlgo {
    var moveIndex = 0
    override fun index(num:Int) {moveIndex=num}
    override fun next(state: GameState): Pair<GameState, SlimState?> {
        return Pair(state.nextMoves[moveIndex],null)
    }

    override fun gameOver() {}
}
