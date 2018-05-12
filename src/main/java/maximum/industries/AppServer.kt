package maximum.industries

import io.ktor.application.*
import io.ktor.features.ContentNegotiation
import io.ktor.request.receive
import io.ktor.response.*
import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.gson.*

data class State(val board: ByteArray,
                 val moveDepth: Int,
                 val whiteMove: Boolean,
                 val moves: Array<Pair<Pair<Int,Int>,Pair<Int,Int>>>,
                 val signature: String)

/*
Example:
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"board": [1,2,3,4,0,0,0,0], "moveDepth": 3, "whiteMove": false, "moves": [{"first":{"first":1, "second":2}, "second":{"first":2, "second":3}}] }' \
     http://localhost:8080/
*/
fun main(args: Array<String>) {
    val gameSpec = loadSpec("chess")

    val server = embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) {
            gson {
                setPrettyPrinting()
            }
        }
        routing {
            post("/") {
                val received = call.receive<State>()
                println(received)
                val state = GameState(gameSpec)
                val wireState = State(state.gameBoard,
                                      10,
                                      true,
                                      arrayOf(Pair(Pair(2,3),Pair(2,4)),
                                              Pair(Pair(3,3),Pair(3,4)),
                                              Pair(Pair(1,1),Pair(2,2))),
                                      "signature")
                call.respond(wireState)
            }
        }
    }
    server.start(wait = true)
}
