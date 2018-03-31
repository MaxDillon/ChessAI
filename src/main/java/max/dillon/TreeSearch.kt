package max.dillon

fun valueFor(player: GameState, node: GameState): Int {
    val sign = if (player.whiteMove == node.whiteMove) 1 else -1
    return sign * node.visitCount
}

fun treeSearch(playerState: GameState): GameState {
    assert(playerState.outcome == GameOutcome.UNDETERMINED)
    var parents = ArrayList<GameState>()
    playerState.expand()

    var maxExpansion = 10000
    while (maxExpansion > 0) {
        parents.clear()
        var currentNode = playerState

        while (true) {
            parents.add(currentNode)
            currentNode = currentNode.nextMoves.maxBy {
                it.score(currentNode.visitCount)
            } ?: throw RuntimeException("wtf")
            if (!currentNode.leaf && currentNode.outcome == GameOutcome.UNDETERMINED) {
                continue
            }
            if (currentNode.outcome != GameOutcome.UNDETERMINED) {
                parents.forEach { it.updateValue(currentNode.totalValue) }
                break
            }
            currentNode.expand()
            maxExpansion--
            if (currentNode.totalValue == 0f) {
                continue
            } else {
                parents.forEach { it.updateValue(currentNode.totalValue) }
                break
            }
        }
    }
    return playerState.nextMoves.maxBy { valueFor(playerState, it) } ?: throw RuntimeException("wtf")
}


fun play(spec: GameGrammar.GameSpec) {
    var state = GameState(spec)
    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
        state = treeSearch(state)
        println(state.description)
        state.printBoardLarge()
    }
}

