package max.dillon

fun treeSearch(state: GameState): GameState {
    state.expand()
    for (i in 1..1600) {
        //println("####################################")
        var node = state
        var stack = ArrayList<GameState>()
        while (!node.leaf && node.gameOutcome() == GameOutcome.UNDETERMINED) {
            stack.add(node)
            node = node.nextMoves.maxBy { it.score(node.visitCount) } ?: throw RuntimeException("wtf")
            //println("${node.description} ${node.leaf}")
        }
        node.expand()
        for (parent in stack) {
            parent.updateValue(node.totalValue * (if (node.whiteMove == parent.whiteMove) 1 else -1))
        }
    }
    return state.nextMoves.maxBy { state.visitCount } ?: throw RuntimeException("wtf")
}


fun play(spec: GameGrammar.GameSpec) {
    var state = GameState(spec)
    while (state.gameOutcome() == GameOutcome.UNDETERMINED) {
        state = treeSearch(state)
        println(state.description)
        state.printBoardLarge()
    }
}

