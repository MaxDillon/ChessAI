package maximum.industries

import maximum.industries.GameGrammar.GameSpec
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.lossfunctions.impl.PseudoSpherical
import org.nd4j.shade.jackson.databind.jsontype.NamedType
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
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
    return if (gameSpec.moveSource == GameGrammar.MoveSource.MOVESOURCE_ENDS) P else N * N
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
                    dropoutRetain: Double = 1.0,
                    norm: Boolean = true) {
        if (norm) queueLayer(BatchNormalization())
        queueLayer(ConvolutionLayer.Builder(kernel, kernel)
                           .nIn(inChannels).nOut(filters)
                           .padding(pad, pad).stride(stride, stride)
                           .activation(activation)
                           .weightInit(init)
                           .dropOut(dropoutRetain)
                           .build())
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
                           .lossFunction(loss.iLossFunction).build())
    }

    fun output(nOut: Int,
               activation: Activation = Activation.TANH,
               loss: LossFunction = LossFunction.L2,
               init: WeightInit = WeightInit.RELU,
               norm: Boolean = true) {
        if (norm) queueLayer(BatchNormalization())
        queueLayer(OutputLayer.Builder().nOut(nOut).activation(activation)
                           .lossFunction(loss.iLossFunction).weightInit(init).build())
    }
}

interface IModel {
    fun newModel(gameSpec: GameGrammar.GameSpec,
                 learningRateOverride: Double? = null,
                 regularizationOverride: Double? = null): ComputationGraph
}

fun modelHas(model: ComputationGraph, layerName: String): Boolean {
    for (layer in model.layers) {
        if (layer.conf().layer.layerName == layerName) return true
    }
    return false
}

fun loadModel(modelName: String): Triple<ComputationGraph, String, Int>? {
    try {
        val model = ModelSerializer.restoreComputationGraph(modelName, false)
        val tokens = modelName.split(".").toMutableList()
        if (tokens.first() == "model") tokens.removeAt(0)
        val batch = tokens.last().toIntOrNull() ?: 0
        if (batch > 0) tokens.removeAt(tokens.lastIndex)
        return Triple(model, tokens.joinToString("."), batch)
    } catch (e: Exception) {
        return null
    }
}

fun newModel(modelName: String, gameSpec: GameSpec,
             learningRateOverride: Double?,
             regularizationOverride: Double?): ComputationGraph? {
    try {
        val factory: IModel =
                Class.forName("maximum.industries.models.$modelName")
                        .newInstance() as IModel
        return factory.newModel(gameSpec, learningRateOverride, regularizationOverride)
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
        |                              will be used to initalize the new one (which will only work
        |                              if the architectures are the same).
        |        [-saveas <name>]      A name pattern for saved models. Default is <game>.<model>
        |                              or a continuation of the pattern from a loaded model.
        |        [-data <datafile>]    Will train on a file matching 'data.<datafile>.done'
        |                              or on recent files matching data.<datafile>.nnnnn.done
        |                              Default is <game>
        |        [-lastn <n>]          The number of recent files to train on. Default is 200.
        |        [-drawweight <w>]     The weight to use for sampling of draws.
        |        [-batch <n>]          The batch size.
        """.trimMargin())
}

fun main(args: Array<String>) {
    // use half precision floats for all tensors
    Nd4j.setDataType(DataBuffer.Type.HALF);
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

//    NeuralNetConfiguration.reinitMapperWithSubtypes(
//            Collections.singletonList(NamedType(PseudoSpherical::class.java)))

    if (args.contains("-h")) {
        return trainUsage()
    }
    val gameName = args[0]
    val gameSpec = loadSpec(gameName)

    val learningRateOverride = getArg(args, "rate")?.toDouble()
    val regularizationOverride = getArg(args, "regu")?.toDouble()

    val from = getArg(args, "from")
    val priorModelInfo = if (from != null) loadModel(from) else null

    val new = getArg(args, "new")
    val newModel = if (new != null) newModel(new, gameSpec,
                                             learningRateOverride,
                                             regularizationOverride) else null

    var defaultSaveAs = ""
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
                    newModel.init(priorModelInfo.first.params(true),
                                  true)
                    defaultSaveAs = priorModelInfo.second
                    startingBatch = priorModelInfo.third
                } else {
                    defaultSaveAs = "$gameName.${new!!}"
                    newModel.init()
                }
                newModel
            }

    if (learningRateOverride != null) {
        model.setLearningRate(learningRateOverride)
    }

    val useValue = modelHas(model, "value")
    val usePolicy = modelHas(model, "policy")
    val useLegal = modelHas(model, "legal")

    val dataPattern = getArg(args, "data") ?: gameName
    val lastN = getArg(args, "lastn")?.toInt() ?: 200
    val saveAs = getArg(args, "saveas") ?: defaultSaveAs
    val drawWeight = getArg(args, "drawweight")?.toDouble() ?: 1.0
    val batchSize = getArg(args, "batch")?.toInt() ?: 200
    val logFile = getArg(args, "logfile") ?: "log.$saveAs"
    var saveevery = getArg(args, "saveevery")?.toInt() ?: 1000
    var updates = getArg(args, "updates")?.toInt() ?: 100
    var batchCount = startingBatch

    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    val trainReader = FileInstanceReader(0.2, drawWeight, lastN, dataPattern, "done")
    val testReader = FileInstanceReader(0.2, drawWeight, lastN, dataPattern, "test")

    var ema_train = 0.0
    var ema_test = 0.0
    fun ema(ema: Double, next: Double, w: Double) =
            if (ema == 0.0) next else w * ema + (1 - w) * next

    while (true) {
        val train_batch = getBatch(gameSpec, trainReader, batchSize, useValue, usePolicy, useLegal)
        if (batchCount % 5 == 1) {
            val test_batch = getBatch(gameSpec, testReader, batchSize, useValue, usePolicy, useLegal)

            val train_score = model.score(train_batch)
            val test_score = model.score(test_batch)

            ema_train = ema(ema_train, train_score, 0.9)
            ema_test = ema(ema_test, test_score, 0.9)

            File(logFile).appendText(
                    "${batchCount},${ema_train.toFloat().f3()},${ema_test.toFloat().f3()}\n")
        }
        model.fit(train_batch)

        batchCount++
        if (batchCount % saveevery == 0) {
            ModelSerializer.writeModel(model, "model.$saveAs.$batchCount", true)
            if (--updates == 0) System.exit(0)
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
                         val lastN: Int, val filePattern: String,
                         val extension: String) : InstanceReader {
    private val rand = Random()
    var currentStream = nextStream()
    val counts = HashMap<Pair<Instance.Player, Int>, Int>()
    var total = 0

    // return a recent file of training data matching 'data.{filepattern}.nnnnnnn.done'
    // or else return just the one file if a full name was given instead of a pattern.
    // we'll be constantly creating new files, and this will pick from the last N
    fun nextStream(): FileInputStream {
        val matcher = BiPredicate<Path, BasicFileAttributes> { file, _ ->
            val fileName = file.fileName.toString()
            fileName.matches(Regex(".*data.$filePattern.[0-9]+.$extension")) ||
            fileName == filePattern
        }
        val paths = ArrayList<Path>()
        for (path in find(Paths.get("."), 1, matcher).iterator()) paths.add(path)
        val recent = paths.sortedByDescending { it.fileName }.subList(0, min(lastN, paths.size))
        if (recent.isEmpty()) throw RuntimeException("No files match")
        return FileInputStream(recent[rand.nextInt(recent.size)].toString())
    }

    override fun next(): Instance.TrainingInstance {
        while (true) {
            val instance = try {
                Instance.TrainingInstance.parseDelimitedFrom(currentStream)
            } catch (_: Exception) {
                null
            }
            if (instance == null) {
                try {
                    currentStream.close()
                    currentStream = nextStream()
                } catch (_: Exception) {
                }
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

data class TrainingData(val input: INDArray, val value: INDArray, val policy: INDArray, val legal: INDArray)

fun initTrainingData(gameSpec: GameSpec, batchSize: Int): TrainingData {
    val sz = gameSpec.boardSize
    val np = gameSpec.pieceCount - 1
    val input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
    val value = Nd4j.zeros(batchSize, 1)
    val policy = Nd4j.zeros(batchSize, policySize(gameSpec))
    val legal = Nd4j.ones(batchSize, policySize(gameSpec)).mul(-1)
    return TrainingData(input, value, policy, legal)
}

fun parseBatch(gameSpec: GameSpec, instances: Array<Instance.TrainingInstance>,
               useValue: Boolean, usePolicy: Boolean, useLegal: Boolean): MultiDataSet {
    val batchSize = instances.size
    val (input, value, policy, legal) = initTrainingData(gameSpec, batchSize)
    for (i in 0 until batchSize) {
        val reflection = rand.nextInt(4)
        instances[i].toBatchTrainingInput(gameSpec, i, reflection, input, value, policy, legal)
    }
    val outputs = ArrayList<INDArray>()
    if (useValue) outputs.add(value)
    if (usePolicy) outputs.add(policy)
    if (useLegal) outputs.add(legal)
    return MultiDataSet(arrayOf(input), outputs.toTypedArray())
}

fun getBatch(gameSpec: GameSpec, reader: InstanceReader, batchSize: Int,
             useValue: Boolean, usePolicy: Boolean, useLegal: Boolean): MultiDataSet {
    val instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances, useValue, usePolicy, useLegal)
}

