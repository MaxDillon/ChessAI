package maximum.industries

import com.google.protobuf.ByteString
import com.google.protobuf.TextFormat
import org.deeplearning4j.util.ModelSerializer
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.attribute.BasicFileAttributes
import java.util.*
import java.util.function.BiPredicate

fun Float.f3(): String = String.format("%.3f", this)

class HumanInput : GameSearchAlgo {
    override fun next(state: GameState): Pair<GameState, SlimState?> {
        fun parse(s: String) = Pair(s.first() - 'a', s.substring(1).toInt() - 1)
        while (true) {
            println("Enter move (e.g., 'a1' to place or 'a1:b1' to move):")
            val str = readLine()
            if (str == null || str == "") continue
            val tokens = str.split(":")
            if (tokens.size > 2) continue
            val (x1, y1) = if (tokens.size == 1) Pair(-1, -1) else parse(tokens[0])
            val (x2, y2) = parse(tokens[tokens.size - 1])
            for (move in state.nextMoves) {
                if (move.x2 == x2 && move.y2 == y2 &&
                    (move.x1 == x1 || x1 < 0) && (move.y1 == y1 || y1 < 0)) {
                    return Pair(move, null)
                }
            }
            println("Try again")
        }
    }
    override fun gameOver() {}
}

fun recordGame(finalState: GameState, slimStates: ArrayList<SlimState>, outputStream: OutputStream) {
    slimStates.forEach { slim ->
        Instance.TrainingInstance.newBuilder().apply {
            boardState = ByteString.copyFrom(slim.state)
            player = slim.player
            gameLength = finalState.moveDepth
            outcome = when (finalState.outcome) {
                Outcome.WIN -> 1
                Outcome.LOSE -> -1
                Outcome.DRAW -> 0
                else -> throw(RuntimeException("undetermined state at end of game"))
            }
            slim.treeSearchResults.forEach {
                addTreeSearchResult(it)
            }
        }.build().writeDelimitedTo(outputStream)
    }
}

fun play(gameSpec: GameGrammar.GameSpec,
         white: GameSearchAlgo, black: GameSearchAlgo,
         stream: OutputStream) {
    var state = GameState(gameSpec)
    val history = ArrayList<SlimState>()

    while (state.outcome == Outcome.UNDETERMINED) {
        println("$state")
        state.printBoard()
        val (next, slim) = if (state.player == Player.WHITE) {
            white.next(state)
        } else {
            black.next(state)
        }
        state = next
        if (slim != null) history.add(slim)
    }
    recordGame(state, history, stream)
    white.gameOver()
    black.gameOver()

    state.printBoard()
    val outcome: Any = when (state.outcome) {
        Outcome.WIN -> if (state.player == Player.WHITE) Player.WHITE else Player.BLACK
        Outcome.LOSE -> if (state.player == Player.BLACK) Player.WHITE else Player.BLACK
        else -> "DRAW"
    }
    println("""|####################################################
               |Outcome: $outcome
               |####################################################
               |""".trimMargin())
}

fun getAlgo(algo: String, iter: Int, exploration: Double, temperature: Double): GameSearchAlgo {
    return when (algo) {
        "mcts" -> MonteCarloTreeSearch(VanillaMctsStrategy(exploration, temperature), iter)
        "human" -> HumanInput()
        else -> {
            val modelName = if (algo.endsWith(".*")) getLatest(algo) else algo
            val model = ModelSerializer.restoreComputationGraph(modelName)
            MonteCarloTreeSearch(AlphaZeroMctsStrategy(model, exploration, temperature), iter)
        }
    }
}

fun loadSpec(game: String): GameGrammar.GameSpec {
    val file = "src/main/data/$game.textproto"
    val specStr = String(Files.readAllBytes(Paths.get(file)))
    return GameGrammar.GameSpec.newBuilder().apply {
        TextFormat.getParser().merge(specStr, this)
        addPieceBuilder(0) // hack: inserting null piece at index 0
    }.build()
}

fun getLatest(modelFile: String): String {
    val matcher = BiPredicate<Path, BasicFileAttributes> { file, _ ->
        file.fileName.toString().matches(Regex(modelFile))
    }
    val paths = java.util.ArrayList<Path>()
    for (path in Files.find(Paths.get("."), 1, matcher).iterator()) paths.add(path)
    return paths.sortedBy { it.fileName }.last().toString()
}

fun appUsage() {
    println("""
        |java PlayKt <game>
        |        [-white <model>]      Model for the white player. Default is mcts.
        |        [-black <model>]      Model for the black player. Default is mcts.
        |        [-n <n>]              The number of games to play. Default=100.
        |        [-iter <iter>]        The number of mcts rollouts to perform.
        |        [-exploration <e>     Governs exploration in tree search. Default=1
        |        [-temperature <t>     Governs move selection exponent. Default=0.2
        |        [-saveas <name>]      A name pattern for saved games. Default is <game>
        |<model> may be 'mcts' to run with Monte Carlo tree search only,
        |            or 'human' to enter moves manually
        |            or a model filename.
        |If the mode filename ends in .* then it will run against the latest version.
        """.trimMargin())
}

fun getArg(args: Array<String>, arg: String): String? {
    for (i in args.indices) {
        if (args[i] == "-$arg" || args[i] == "--$arg") return args[i + 1]
    }
    return null
}

fun main(args: Array<String>) {
    if (args.contains("-h")) {
        return appUsage()
    }
    val game = args[0]
    val white = getArg(args, "white") ?: "mcts"
    val black = getArg(args, "black") ?: "mcts"
    val n = getArg(args, "n")?.toInt() ?: 100
    val iter = getArg(args, "iter")?.toInt() ?: 1600
    val temperature = getArg(args, "temperature")?.toDouble() ?: 0.2
    val exploration = getArg(args, "exploration")?.toDouble() ?: 1.0
    val saveas = getArg(args, "saveas") ?: game
    val baseName = "data.$saveas.${System.currentTimeMillis()}"
    val workFile = "$baseName.work"
    val doneFile = "$baseName.done"
    val outputStream = FileOutputStream(workFile)

    val gameSpec = loadSpec(game)

    val whiteAlgo = getAlgo(white, iter, exploration, temperature)
    val blackAlgo =
            if (white == black) whiteAlgo
            else getAlgo(black, iter, exploration, temperature)

    for (i in 1..n) play(gameSpec, whiteAlgo, blackAlgo, outputStream)
    Files.move(Paths.get(workFile), Paths.get(doneFile))
}
