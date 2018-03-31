package max.dillon

fun valueFor(player: GameState, node: GameState): Float {
    val sign = if (player.whiteMove == node.whiteMove) 1 else -1
    return sign * node.totalValue
}

fun Float.f3(): String = String.format("%.3f", this)

fun treeSearch(playerState: GameState): GameState {
    assert(playerState.outcome == GameOutcome.UNDETERMINED)
    var parents = ArrayList<GameState>()
    playerState.expand()
    if (playerState.nextMoves.size == 1) {
        return playerState.nextMoves[0]
    }

    var maxExpansion = 5000
    while (maxExpansion > 0) {
        parents.clear()
        var currentNode = playerState

        while (true) {
            parents.add(currentNode)
            currentNode = currentNode.nextMoves.maxBy {
                it.scoreFor(currentNode)
            } ?: throw RuntimeException("wtf")
            if (!currentNode.leaf && currentNode.outcome == GameOutcome.UNDETERMINED) {
                continue
            }
            maxExpansion--
            currentNode.expand() // make sure we have node value evaluated
            if (currentNode.outcome != GameOutcome.UNDETERMINED) {
                parents.forEach { it.updateValue(valueFor(it, currentNode)) }
                break
            }
            if (currentNode.totalValue == 0f) {
                continue
            } else {
                parents.forEach { it.updateValue(valueFor(it, currentNode)) }
                break
            }
        }
    }
    return playerState.nextMoves.maxBy {
        print("${it.description} ${it.visitCount} ")
        println("${(valueFor(playerState, it) / it.visitCount).f3()} ${it.prior.f3()}")
        it.visitCount
    } ?: throw RuntimeException("wtf")
}


fun play(spec: GameGrammar.GameSpec) {
    var state = GameState(spec)
    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
        state = treeSearch(state)
        println(state.description)
        state.printBoardLarge()
    }
}

