package maximum.industries

import com.squareup.moshi.JsonAdapter
import com.squareup.moshi.Moshi



class Server {
    val moshi = Moshi.Builder().build()
    val adapter = moshi.adapter<StateMessage>(StateMessage::class.java)

    fun stateMessageToJason(state: StateMessage) = adapter.toJson(state)
    fun jsonToStateMessage(json: String) = adapter.fromJson(json)
}


data class StateMessage(val state: ByteArray,
                        val moves: Map<Pair<Byte,Byte>,Pair<Byte,Byte>>,
                        val isWhite: Boolean) {

}

fun GameState.toStateMessage(): StateMessage {
    val size = gameSpec.boardSize

    val array = ByteArray(size * size) {
        val (x, y) = gameSpec.indexToXy(it)
        at(x, y).toByte()

    }
    val moves = mapOf(*this.nextMoves.map { (it._x1 to it._y1) to (it._x2 to it._y2) }.toTypedArray())
    val isWhite = player.eq(Player.WHITE)

    return StateMessage(array,moves,isWhite)
}


