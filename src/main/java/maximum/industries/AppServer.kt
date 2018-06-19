package maximum.industries

import io.ktor.application.call
import io.ktor.application.install
import io.ktor.content.default
import io.ktor.content.files
import io.ktor.content.static
import io.ktor.features.ContentNegotiation
import io.ktor.gson.gson
import io.ktor.request.receive
import io.ktor.response.respond
import io.ktor.routing.get
import io.ktor.routing.post
import io.ktor.routing.routing
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty

/*
Example:
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"board": [1,2,3,4,0,0,0,0], "moveDepth": 3, "whiteMove": false, "moves": [{"first":{"first":1, "second":2}, "second":{"first":2, "second":3}}] }' \
     http://localhost:8080/
*/
fun main(args: Array<String>) {
    val gameSpec = loadSpec("chess2")
    var state = newGame(gameSpec)

    val server = embeddedServer(Netty, port = 8080) {
        var white = getAlgo("mcts", SearchParameters(iterations=1, exploration=1.0, temperature=0.3, rampBy = 1))
        var black = getAlgo("mcts", SearchParameters(iterations=1, exploration=1.0, temperature=0.3, rampBy = 1))
        var playerWhite = true

        install(ContentNegotiation) {
            gson {
                setPrettyPrinting()
            }
        }
        routing {

            post("/start") {
                playerWhite = call.receive<Boolean>()
                state = newGame(gameSpec)
                val whiteAlgo = if (playerWhite) "gui" else "model2:prod_model.chess2"
                white = getAlgo(whiteAlgo, SearchParameters(iterations = 2000,
                                                            exploration = 0.3,
                                                            temperature = 0.1,
                                                            rampBy = 0,
                                                            priority_exponent = 2.0,
                                                            priority_uniform = 1.0))

                val blackAlgo = if (playerWhite) "model2:prod_model.chess2" else "gui"
                black = getAlgo(blackAlgo, SearchParameters(iterations = 2000,
                                                            exploration = 0.3,
                                                            temperature = 0.1,
                                                            rampBy = 0,
                                                            priority_exponent = 2.0,
                                                            priority_uniform = 1.0))
                call.respond(Pair(state.toWireState(), gameSpec))
            }

            get("/opponentMove") {
                state = if (playerWhite) black.next(state).first else white.next(state).first
                call.respond(state.toWireState())
            }

            post("/move") {
                val received = call.receive<WireState>()
                val color = if (playerWhite) white else black
                if (true) {
                    color.index(received.moveIndex)
                    state = color.next(received.toGameState(gameSpec)).first

                    call.respond(state.toWireState())
                } else {
                    call.respond(received)
                }

            }

            static {
                default("static/webpage.html")
                files("static")
            }

        }
    }
    server.start(wait = true)
}

data class ObjectState(val board: ByteArray,
                       val moves: IntArray,
                       val whiteMove: Boolean,
                       val moveDepth: Int)

data class WireState(val state: ObjectState,
                     val moveIndex: Int,
                     val signature: Int)

class GuiInput : GameSearchAlgo {
    var moveIndex = 0
    override fun index(num: Int) {
        moveIndex = num
    }

    override fun next(state: GameState): Pair<GameState, SlimState?> {
        return Pair(state.nextMoves[moveIndex], null)
    }

    override fun gameOver() {}
}
