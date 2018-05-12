package maximum.industries

import io.ktor.application.*
import io.ktor.features.ContentNegotiation
import io.ktor.http.*
import io.ktor.request.receive
import io.ktor.response.*
import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.gson.*

data class Foo(val x: String, val y: Int)

fun main(args: Array<String>) {
    val server = embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) {
            gson {
                setPrettyPrinting()
            }
        }
        routing {
            post("/") {
                val foo = call.receive<Foo>()
                call.respondText(foo.toString(), ContentType.Text.Plain)
            }
        }
    }
    server.start(wait = true)
}
