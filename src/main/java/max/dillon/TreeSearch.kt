package max.dillon


fun predict(state: GameState): Pair<Float, FloatArray> {


    TODO("Implement")
}


fun subset(array: FloatArray, states: ArrayList<GameState>): FloatArray {


    TODO("Implement")
}


fun expand(state: GameState) {
    state.nextMoves = state.getLegalNextStates()
    val (value, priors) = predict(state)
    for (move in state.nextMoves) {
        move.prior = priors[move.getIndex()]
    }


}

fun treeSearch(state: GameState): Array<TreeSearchResult> {
    val legalMoves = state.getLegalNextStates()
    val predictions = predict(state)


    TODO("Implement")
}


data class TreeSearchResult(
        val state: GameState,
        val prior: Float,
        var count: Int,
        var value: Float
)