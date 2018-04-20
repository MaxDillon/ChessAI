package maximum.industries


import maximum.industries.*
import maximum.industries.FileInstanceReader
import maximum.industries.GameGrammar.*
import maximum.industries.InstanceReader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex
import org.deeplearning4j.nn.conf.inputs.InputType
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
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.FileInputStream
import java.io.InputStream
import java.nio.file.Files.find
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.attribute.BasicFileAttributes
import java.util.*
import java.util.function.BiPredicate
import kotlin.math.min

fun policyChannels(gameSpec: GameSpec): Int {
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    return if (gameSpec.moveSource == GameGrammar.MoveSource.ENDS) P else N * N
}

fun policySize(gameSpec: GameSpec): Int {
    val N = gameSpec.boardSize
    return policyChannels(gameSpec) * N * N
}

fun ComputationGraphConfiguration.GraphBuilder.addResidual(input: String,
                                                           output: String,
                                                           convFilters: Int):
        ComputationGraphConfiguration.GraphBuilder { this
            .addLayer("${output}_conv1",
            ConvolutionLayer.Builder(3, 3)
                    .nOut(convFilters).stride(1, 1).padding(1, 1)
                    .activation(Activation.RELU).weightInit(WeightInit.RELU)
                    .build(), input)
            .addLayer("${output}_conv2",
                      ConvolutionLayer.Builder(3, 3)
                              .nOut(convFilters)
                              .stride(1, 1)
                              .padding(1, 1)
                              .activation(Activation.RELU)
                              .weightInit(WeightInit.RELU)
                              .build(), "${output}_conv1")
            .addLayer("${output}_batch",
                      BatchNormalization(), "${output}_conv2")
            .addVertex("$output",
                    ElementWiseVertex(ElementWiseVertex.Op.Add), input, "${output}_batch")
    return this
}

fun newModel(gameSpec: GameSpec, N: Int, P: Int, rate: Double): ComputationGraph {
    val inChannels = 2 * P + 1
    val convFilters = 48
    val mconfig = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(rate)
            .regularization(true).l2(1e-6)
            .updater(Updater.NESTEROVS)
            .weightInit(WeightInit.RELU)
            .graphBuilder()
            .addInputs("input")
            .setInputTypes(InputType.convolutional(N, N, inChannels))

            // =======================================================
            // Convolutional Layers, Batch Normalization, Skips
            // =======================================================

            .addLayer("conv1", ConvolutionLayer.Builder(3, 3)
                    .nIn(inChannels).nOut(convFilters)
                    .stride(1, 1).padding(1, 1)
                    .activation(Activation.RELU)
                    .build(), "input")
            .addLayer("norm_conv1", BatchNormalization(), "conv1")

            .addResidual("norm_conv1", "residual1", convFilters)
            .addResidual("residual1", "residual2", convFilters)

            // =======================================================
            // Convolutional Output Reshaping
            // =======================================================

            .addLayer("legal_conv", ConvolutionLayer.Builder(3, 3)
                    .nOut(policyChannels(gameSpec))
                    .stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .build(), "residual2")
            .addVertex("legal_reshape",
                       PreprocessorVertex(
                               CnnToFeedForwardPreProcessor(N, N, policyChannels(gameSpec))),
                       "legal_conv")
            .addLayer("legal_batch", BatchNormalization(), "legal_reshape")
            .addLayer("legal",
                      LossLayer.Builder(LossFunctions.LossFunction.L2)
                              .activation(Activation.TANH)
                              .build(),
                      "legal_batch")

            .addLayer("policy_conv", ConvolutionLayer.Builder(3, 3)
                    .nOut(policyChannels(gameSpec))
                    .stride(1, 1).padding(1, 1)
                    .activation(Activation.LEAKYRELU)
                    .build(), "residual2")
            .addVertex("policy_reshape",
                       PreprocessorVertex(
                               CnnToFeedForwardPreProcessor(N, N, policyChannels(gameSpec))),
                       "policy_conv")
            .addLayer("policy_batch", BatchNormalization(), "policy_reshape")
            .addLayer("policy",
                      LossLayer.Builder(LossFunctions.LossFunction.KL_DIVERGENCE)
                              .activation(Activation.SOFTMAX)
                              .build(),
                      "policy_batch")

            // =======================================================
            // Value Output
            // =======================================================

            .addLayer("dense", DenseLayer.Builder()
                    .nOut(32)
                    .activation(Activation.LEAKYRELU)
                    .build(), "residual2")
            .addLayer("dense_batch", BatchNormalization(), "dense")
            .addLayer("value", OutputLayer.Builder(LossFunctions.LossFunction.L2)
                    .nOut(1)
                    .activation(Activation.TANH)
                    .build(), "dense_batch")

            .setOutputs( "value", "policy", "legal")
            .build()
    val model = ComputationGraph(mconfig)
    model.init()
    return model
}

fun trainUsage() {
    println("""
        |java TrainingKt <game>
        |        [-data <datafile>]    Will train on a file matching 'data.<datafile>.done'
        |                              or on recent files matching data.<datafile>.nnnnn.done
        |                              Default is <game>
        |        [-lastn <n>]          The number of recent files to train on. Default is 10.
        |        [-model <model>]      An optional saved model to continue training.
        |        [-saveas <name>]      A name pattern for saved models. Default is <game>
        |        [-rate <rate>]        Learning rate (for new models)
        """.trimMargin())
}

fun main(args: Array<String>) {
    if (args.contains("-h")) {
        return trainUsage()
    }
    val game = args[0]
    val filePattern = getArg(args, "data") ?: game
    val lastN = getArg(args, "lastn")?.toInt() ?: 100
    val modelName = getArg(args, "model")
    val outName = getArg(args, "saveas") ?: game
    val rate = getArg(args, "rate")?.toDouble() ?: 0.0001

    val gameSpec = loadSpec(game)
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    val batchSize = 1000

    val model = if (modelName == null) {
        newModel(gameSpec, N, P, rate)
    } else {
        ModelSerializer.restoreComputationGraph(modelName)
    }

    //model.setListeners(ScoreIterationListener())
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    var batchCount = 0
    val reader = FileInstanceReader(0.2, lastN, filePattern)
    while (true) {
        model.fit(getBatch(gameSpec, reader, batchSize))
        batchCount++
        if (batchCount % 1000 == 0) {
            ModelSerializer.writeModel(model, "model.$outName.$batchCount", true)
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

class FileInstanceReader(val prob: Double, val lastN: Int, val filePattern: String) : InstanceReader {
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

    override fun next(): Instance.TrainingInstance {
        while (true) {
            val instance = Instance.TrainingInstance.parseDelimitedFrom(instream)
            if (instance == null) {
                instream.close()
                instream = nextStream()
                continue
            } else if (rand.nextDouble() < prob) {
                return instance
            }
        }
    }
}

fun parseBatch(gameSpec: GameSpec, instances: Array<Instance.TrainingInstance>): MultiDataSet {
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
    return MultiDataSet(arrayOf(input), arrayOf(value, policy, legal))
}

fun getBatch(gameSpec: GameSpec, reader: InstanceReader, batchSize: Int): MultiDataSet {
    val instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances)
}

