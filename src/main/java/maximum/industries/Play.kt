package maximum.industries

import com.google.protobuf.ByteString
import com.google.protobuf.TextFormat
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.impl.PseudoSpherical
import org.nd4j.shade.jackson.databind.jsontype.NamedType
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.attribute.BasicFileAttributes
import java.util.*
import java.util.function.BiPredicate

fun Float.f1(): String = String.format("%.1f", this)
fun Float.f3(): String = String.format("%.3f", this)

interface GameSearchAlgo {
    fun next(state: GameState): Pair<GameState, SlimState?>
    fun gameOver()
}

fun play(gameSpec: GameGrammar.GameSpec,
         white: GameSearchAlgo, black: GameSearchAlgo,
         stream: OutputStream) {
    var state = GameState(gameSpec)
    val history = ArrayList<SlimState>()

    while (state.outcome == Outcome.UNDETERMINED) {
        println("$state")
        state.printBoard()
        val (next, slim) = if (state.player.eq(Player.WHITE)) {
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
        Outcome.WIN -> if (state.player.eq(Player.WHITE)) Player.WHITE else Player.BLACK
        Outcome.LOSE -> if (state.player.eq(Player.BLACK)) Player.WHITE else Player.BLACK
        else -> "DRAW"
    }
    println("""|####################################################
               |Outcome: $outcome
               |####################################################
               |""".trimMargin())
}

fun recordGame(finalState: GameState, slimStates: ArrayList<SlimState>, outputStream: OutputStream) {
    slimStates.forEach { slim ->
        val slimWhite = slim.player.eq(Instance.Player.WHITE)
        val finalWhite = finalState.player.eq(Player.WHITE)
        Instance.TrainingInstance.newBuilder().apply {
            boardState = ByteString.copyFrom(slim.state)
            player = slim.player
            gameLength = finalState.moveDepth
            outcome = when (finalState.outcome) {
                Outcome.WIN -> if (slimWhite == finalWhite) 1 else -1
                Outcome.LOSE -> if (slimWhite == finalWhite) -1 else 1
                Outcome.DRAW -> 0
                else -> throw(RuntimeException("undetermined state at end of game"))
            }
            slim.treeSearchResults.forEach {
                addTreeSearchResult(it)
            }
        }.build().writeDelimitedTo(outputStream)
    }
}

// A strategy allowing a human to play against an algorithm
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

fun getAlgo(game: String, algo: String,
            iter: Int, exploration: Double, temperature: Double): GameSearchAlgo {
    val toks = algo.split(":")
    return when (toks[0]) {
        "mcts" -> MonteCarloTreeSearch(VanillaMctsStrategy(
                exploration, temperature), iter)
        "dmcts" -> MonteCarloTreeSearch(DirichletMctsStrategy(
                exploration, temperature, floatArrayOf(1.0f, 0.0f, 0.5f)), iter)
        "model" -> {
            val modelName = if (toks[1].endsWith(".*")) getLatest(toks[1]) else toks[1]
            NeuralNetConfiguration.reinitMapperWithSubtypes(
                    Collections.singletonList(NamedType(PseudoSpherical::class.java)))
            val model = ModelSerializer.restoreComputationGraph(modelName)
            MonteCarloTreeSearch(AlphaZeroMctsStrategy(model, exploration, temperature), iter)
        }
        "human" -> HumanInput()
        else -> throw RuntimeException("no algo specified")
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
        |        [-white <model>]    Model for the white player. Default is mcts.
        |        [-black <model>]    Model for the black player. Default is mcts.
        |        [-n <n>]            The number of games to play. Default=100.
        |        [-witer <iter>]     The number of rollouts/evals to perform for white.
        |        [-biter <iter>]     The number of rollouts/evals to perform for black.
        |        [-wexpl <e>         Governs exploration in tree search for white. Default=0.5
        |        [-bexpl <e>         Governs exploration in tree search for black. Default=0.5
        |        [-wtemp <t>         Governs move selection exponent for white. Default=0.1
        |        [-btemp <t>         Governs move selection exponent for black. Default=0.1
        |        [-saveas <name>]    A name pattern for saved games. Default is <game>
        |<model> may be 'mcts' to run with Monte Carlo tree search only,
        |            or 'dmcts' to run with Dirichlet Monte Carlo tree search,
        |            or 'human' to enter moves manually
        |            or a model filename.
        |If the mode filename ends in .* then it will run against the latest version.
        """.trimMargin())
}

fun getArg(args: Array<String>, arg: String): String? {
    for (i in args.indices) {
        if (args[i] == "-$arg" || args[i] == "--$arg") {
            println("Using: -$arg = ${args[i+1]}")
            return args[i + 1]
        }
    }
    return null
}

fun main(args: Array<String>) {
    if (args.contains("-h")) {
        return appUsage()
    }
    val game = args[0]
    val n = getArg(args, "n")?.toInt() ?: 100
    val white = getArg(args, "white") ?: "mcts"
    val black = getArg(args, "black") ?: "mcts"
    val witer = getArg(args, "witer")?.toInt() ?: 100
    val biter = getArg(args, "biter")?.toInt() ?: 100
    val wtemp = getArg(args, "wtemp")?.toDouble() ?: 0.1
    val btemp = getArg(args, "btemp")?.toDouble() ?: 0.1
    val wexpl = getArg(args, "wexpl")?.toDouble() ?: 0.5
    val bexpl = getArg(args, "bexpl")?.toDouble() ?: 0.5
    val saveas = getArg(args, "saveas") ?: game
    val baseName = "data.$saveas.${System.currentTimeMillis()}"
    val workFile = "$baseName.work"
    val doneFile = "$baseName.done"

    println("Log file: $workFile")
    val outputStream = FileOutputStream(workFile)

    val gameSpec = loadSpec(game)

    val whiteAlgo = getAlgo(game, white, witer, wexpl, wtemp)
    val blackAlgo =
            if (white == black) whiteAlgo
            else getAlgo(game, black, biter, bexpl, btemp)

    for (i in 1..n) play(gameSpec, whiteAlgo, blackAlgo, outputStream)
    Files.move(Paths.get(workFile), Paths.get(doneFile))
}
