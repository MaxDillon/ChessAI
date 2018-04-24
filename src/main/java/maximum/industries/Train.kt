package maximum.industries

import maximum.industries.GameGrammar.GameSpec
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.io.*
import java.nio.file.Files.find
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.attribute.BasicFileAttributes
import java.util.*
import java.util.function.BiPredicate
import kotlin.collections.HashMap
import kotlin.math.min

fun inputChannels(gameSpec: GameSpec): Int {
    val P = gameSpec.pieceCount - 1
    return 2 * P + 1
}

fun policyChannels(gameSpec: GameSpec): Int {
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    return if (gameSpec.moveSource == GameGrammar.MoveSource.ENDS) P else N * N
}

fun policySize(gameSpec: GameSpec): Int {
    val N = gameSpec.boardSize
    return policyChannels(gameSpec) * N * N
}

typealias GBuilder = ComputationGraphConfiguration.GraphBuilder

fun GBuilder.F(fn: String, vararg of: String,
               makeLayers: LayerQueue.() -> Unit): GBuilder {
    val queue = LayerQueue()
    queue.makeLayers()
    var inputs = of.toList()
    for (i in 0 until queue.layers.size) {
        val name = if (i == queue.layers.size - 1) fn else "${fn}_${i}"
        println("Adding $name = f(${inputs})")
        addLayer(name, queue.layers[i], *inputs.toTypedArray())
        inputs = listOf(name)
    }
    return this
}

fun GBuilder.R(fn: String, vararg of: String,
               makeLayers: LayerQueue.() -> Unit): GBuilder {
    F("res_$fn", *of, makeLayers = makeLayers)
    val mergeInputs = of.toMutableList()
    mergeInputs.add("res_$fn")
    println("Adding $fn = f(${mergeInputs})")
    addVertex(fn, ElementWiseVertex(ElementWiseVertex.Op.Add), *mergeInputs.toTypedArray())
    return this
}

fun GBuilder.CNN2FF(fn: String, of: String, sz: Int, channels: Int): GBuilder {
    println("Adding $fn = f($of)")
    addVertex(fn, PreprocessorVertex(CnnToFeedForwardPreProcessor(sz, sz, channels)), of)
    return this
}

class LayerQueue {
    val layers = ArrayList<Layer>()
    fun queueLayer(layer: Layer) {
        layers.add(layer)
    }

    fun convolution(inChannels: Int = 0, filters: Int,
                    kernel: Int = 3, pad: Int = 1, stride: Int = 1,
                    activation: Activation = Activation.RELU,
                    init: WeightInit = WeightInit.RELU,
                    norm: Boolean = true) {
        if (norm) queueLayer(BatchNormalization())
        queueLayer(ConvolutionLayer.Builder(kernel, kernel)
                           .nIn(inChannels).nOut(filters)
                           .padding(pad, pad).stride(stride, stride)
                           .activation(activation).weightInit(init).build())
    }

    fun dense(units: Int,
              activation: Activation = Activation.RELU,
              init: WeightInit = WeightInit.RELU,
              norm: Boolean = true) {
        if (norm) queueLayer(BatchNormalization())
        queueLayer(DenseLayer.Builder().nOut(units).activation(activation)
                           .weightInit(init).build())
    }

    fun loss(activation: Activation = Activation.TANH,
             loss: LossFunction = LossFunction.L2) {
        queueLayer(LossLayer.Builder().activation(activation)
                           .lossFunction(loss).build())
    }

    fun output(nOut: Int,
               activation: Activation = Activation.TANH,
               loss: LossFunction = LossFunction.L2,
               init: WeightInit = WeightInit.RELU,
               norm: Boolean = true) {
        if (norm) queueLayer(BatchNormalization())
        queueLayer(OutputLayer.Builder().nOut(nOut).activation(activation)
                           .lossFunction(loss).weightInit(init).build())
    }
}

interface IModel {
    fun newModel(gameSpec: GameGrammar.GameSpec): ComputationGraph
}

fun hasLegal(model: ComputationGraph): Boolean {
    for (layer in model.layers) {
        if (layer.conf().layer.layerName == "legal") return true
    }
    return false
}

fun loadModel(modelName: String): Triple<ComputationGraph, String, Int>? {
    try {
        val model = ModelSerializer.restoreComputationGraph(modelName)
        val tokens = modelName.split(".").toMutableList()
        if (tokens.first() == "model") tokens.removeAt(0)
        val batch = tokens.last().toIntOrNull() ?: 0
        if (batch > 0) tokens.removeAt(tokens.lastIndex)
        return Triple(model, tokens.joinToString("."), batch)
    } catch (e: Exception) {
        return null
    }
}

fun newModel(modelName: String, gameSpec: GameSpec): ComputationGraph? {
    try {
        val factory: IModel =
                Class.forName("maximum.industries.models.$modelName")
                        .newInstance() as IModel
        return factory.newModel(gameSpec)
    } catch (e: Exception) {
        return null
    }
}

fun trainUsage() {
    println("""
        |java TrainingKt <game>
        |        [-new <class>]        An unqualified class name in the maximum.industries.models
        |                              package identifying a factory to construct a new model.
        |        [-from <model>]       An existing model to continue training. If a -new argument
        |                              is also provided, then the weights from the existing model
        |                              wil be used to initalize the new one (which will only work
        |                              if the architectures are the same).
        |        [-saveas <name>]      A name pattern for saved models. Default is <game>.<model>
        |                              or a continuation of the pattern from a loaded model.
        |        [-data <datafile>]    Will train on a file matching 'data.<datafile>.done'
        |                              or on recent files matching data.<datafile>.nnnnn.done
        |                              Default is <game>
        |        [-lastn <n>]          The number of recent files to train on. Default is 200.
        """.trimMargin())
}

fun main(args: Array<String>) {
    if (args.contains("-h")) {
        return trainUsage()
    }
    val gameName = args[0]
    val gameSpec = loadSpec(gameName)

    val from = getArg(args, "from")
    val priorModelInfo = if (from != null) loadModel(from) else null

    val new = getArg(args, "new")
    val newModel = if (new != null) newModel(new, gameSpec) else null

    var defaultSaveAs = gameName
    var startingBatch = 0
    val model =
            if (newModel == null) {
                if (priorModelInfo == null) {
                    println("Error: no model specified or could not load model")
                    return
                } else {
                    defaultSaveAs = priorModelInfo.second
                    startingBatch = priorModelInfo.third
                    priorModelInfo.first
                }
            } else {
                if (priorModelInfo != null) {
                    newModel.init(priorModelInfo.first.params(), true)
                    startingBatch = priorModelInfo.third
                }
                defaultSaveAs = "$gameName.${new!!}"
                newModel
            }

    val useLegal = hasLegal(model)

    val dataPattern = getArg(args, "data") ?: gameName
    val lastN = getArg(args, "lastn")?.toInt() ?: 200
    val saveAs = getArg(args, "saveas") ?: defaultSaveAs
    val drawWeight = getArg(args, "drawweight")?.toDouble() ?: 1.0

    val batchSize = 200
    var batchCount = startingBatch

    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    model.params()
    model.init(model.params(), true)

    val reader = FileInstanceReader(0.2, drawWeight, lastN, dataPattern)
    while (true) {
        model.fit(getBatch(gameSpec, reader, batchSize, useLegal))
        batchCount++
        if (batchCount % 1000 == 0) {
            ModelSerializer.writeModel(model, "model.$saveAs.$batchCount", true)
        }
    }
}

interface InstanceReader {
    fun next(): Instance.TrainingInstance
}

class StreamInstanceReader(val stream: InputStream) : InstanceReader {
    override fun next(): Instance.TrainingInstance {
        return Instance.TrainingInstance.parseDelimitedFrom(stream)
    }
}

class FileInstanceReader(val prob: Double, val drawWeight: Double,
                         val lastN: Int, val filePattern: String) : InstanceReader {
    private val rand = Random()

    // return a recent file of training data matching 'data.{filepattern}.nnnnnnn.done'
    // or else return just the one file if a full name was given instead of a pattern.
    // we'll be constantly creating new files, and this will pick from the last N
    fun nextStream(): FileInputStream {
        val matcher = BiPredicate<Path, BasicFileAttributes> { file, _ ->
            val fileName = file.fileName.toString()
            fileName.matches(Regex(".*data.$filePattern.[0-9]+.done")) ||
            fileName == filePattern
        }
        val paths = ArrayList<Path>()
        for (path in find(Paths.get("."), 1, matcher).iterator()) paths.add(path)
        val recent = paths.sortedByDescending { it.fileName }.subList(0, min(lastN, paths.size))
        if (recent.isEmpty()) throw RuntimeException("No files match")
        return FileInputStream(recent[rand.nextInt(recent.size)].toString())
    }

    var instream = nextStream()
    val counts = HashMap<Pair<Instance.Player, Int>, Int>()
    var total = 0

    override fun next(): Instance.TrainingInstance {
        while (true) {
            val instance = Instance.TrainingInstance.parseDelimitedFrom(instream)
            if (instance == null) {
                instream.close()
                instream = nextStream()
                continue
            } else {
                // adjust probability of selecting an instance so we come out approximately
                // balanced between wins and losses for white and black
                val key = Pair(instance.player, instance.outcome)
                val count = counts.getOrPut(key, { 0 })
                val threshold = prob *
                                if (instance.outcome == 0) drawWeight
                                else if ((count + 1.0) / (total + 1.0) > 0.25) 0.1 else 1.0
                if (rand.nextDouble() < threshold) {
                    counts.put(key, count + 1)
                    if (instance.outcome != 0) total += 1
                    if (total % 5000 == 0) {
                        for (k in counts.keys) print("$k: ${counts[k]}   ")
                        println()
                    }
                    return instance
                }
            }
        }
    }
}

fun parseBatch(gameSpec: GameSpec,
               instances: Array<Instance.TrainingInstance>,
               useLegal: Boolean): MultiDataSet {
    val sz = gameSpec.boardSize
    val np = gameSpec.pieceCount - 1
    val batchSize = instances.size
    val input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
    val value = Nd4j.zeros(batchSize, 1)
    val policy = Nd4j.zeros(batchSize, policySize(gameSpec))
    val legal = Nd4j.ones(batchSize, policySize(gameSpec)).mul(-1)

    for (i in 0 until batchSize) {
        for (j in 0 until instances[i].boardState.size()) {
            val x = j / sz
            val y = j % sz
            val p = instances[i].boardState.byteAt(j).toInt()
            if (p != 0) {
                val channel = if (p > 0) p else np - p
                input.putScalar(intArrayOf(i, channel, x, y), 1)
            }
        }
        val turn = if (instances[i].player.eq(Instance.Player.WHITE)) 1 else -1
        input.put(arrayOf(NDArrayIndex.point(i), NDArrayIndex.point(0),
                          NDArrayIndex.all(), NDArrayIndex.all()), turn)

        for (j in 0 until instances[i].treeSearchResultCount) {
            val tsr = instances[i].treeSearchResultList[j]
            policy.putScalar(intArrayOf(i, tsr.index), tsr.prob)
            legal.putScalar(intArrayOf(i, tsr.index), 1f)
        }
        value.putScalar(i, instances[i].outcome)
    }
    if (useLegal) {
        return MultiDataSet(arrayOf(input), arrayOf(value, policy, legal))
    } else {
        return MultiDataSet(arrayOf(input), arrayOf(value, policy))
    }
}

fun getBatch(gameSpec: GameSpec, reader: InstanceReader,
             batchSize: Int, useLegal: Boolean): MultiDataSet {
    val instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances, useLegal)
}

