//package max.dillon
//
//
//import maximum.industries.GameGrammar
//import maximum.industries.GameGrammar.GameSpec
//import maximum.industries.Instance
//import maximum.industries.loadSpec
//import org.deeplearning4j.nn.api.OptimizationAlgorithm
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration
//import org.deeplearning4j.nn.conf.Updater
//import org.deeplearning4j.nn.conf.inputs.InputType
//import org.deeplearning4j.nn.conf.layers.*
//import org.deeplearning4j.nn.graph.ComputationGraph
//import org.deeplearning4j.nn.weights.WeightInit
//import org.deeplearning4j.ui.api.UIServer
//import org.deeplearning4j.ui.stats.StatsListener
//import org.deeplearning4j.ui.storage.InMemoryStatsStorage
//import org.deeplearning4j.util.ModelSerializer
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.dataset.MultiDataSet
//import org.nd4j.linalg.factory.Nd4j
//import org.nd4j.linalg.lossfunctions.LossFunctions
//import java.io.FileInputStream
//import java.io.InputStream
//import java.nio.file.Files.find
//import java.nio.file.Path
//import java.nio.file.Paths
//import java.nio.file.attribute.BasicFileAttributes
//import java.util.*
//import java.util.function.BiPredicate
//import kotlin.math.min
//
//fun policySize(gameSpec: GameSpec): Int {
//    val N = gameSpec.boardSize
//    val P = gameSpec.pieceCount - 1
//    val src_dim = (
//            if (gameSpec.moveSource == GameGrammar.MoveSource.ENDS) P
//            else N * N)
//    val dst_dim = N * N
//    return src_dim * dst_dim
//}
//
//fun newModel(gameSpec: GameSpec, N: Int, P: Int, rate: Double): ComputationGraph {
//    val inChannels = 2 * P + 1
//    val mconfig = NeuralNetConfiguration.Builder()
//            .learningRate(rate)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .updater(Updater.NESTEROVS)
//            .regularization(true).l2(1e-6)
//            .weightInit(WeightInit.XAVIER)
//            .graphBuilder()
//            .addInputs("input")
//            .setInputTypes(InputType.convolutional(N, N, inChannels))
//
//            .addLayer("conv1", ConvolutionLayer.Builder(3, 3)
//                    .nIn(inChannels).nOut(64)
//                    .stride(1, 1).padding(1, 1)
//                    .activation(Activation.RELU)
//                    .build(), "input")
//            .addLayer("norm_conv1", BatchNormalization(), "conv1")
//
//            .addLayer("conv2", ConvolutionLayer.Builder(3, 3)
//                    .nOut(64)
//                    .stride(1, 1).padding(1, 1)
//                    .activation(Activation.RELU)
//                    .build(), "norm_conv1", "input")
//            .addLayer("norm_conv2", BatchNormalization(), "conv2")
//
//            .addLayer("conv3", ConvolutionLayer.Builder(3, 3)
//                    .nOut(64)
//                    .stride(1, 1).padding(1, 1)
//                    .activation(Activation.RELU)
//                    .build(), "norm_conv2", "norm_conv1")
//            .addLayer("norm_conv3", BatchNormalization(), "conv3")
//
//            .addLayer("dense1", DenseLayer.Builder()
//                    .nOut(50)
//                    .activation(Activation.RELU)
//                    .build(), "norm_conv3", "norm_conv2")
//            .addLayer("norm_dense1", BatchNormalization(), "dense1")
//            .addLayer("dense2", DenseLayer.Builder()
//                    .nOut(40)
//                    .activation(Activation.RELU)
//                    .build(), "norm_dense1")
//            .addLayer("norm_dense2", BatchNormalization(), "dense2")
//            .addLayer("dense3", DenseLayer.Builder()
//                    .nOut(30)
//                    .activation(Activation.RELU)
//                    .build(), "norm_dense2")
//
//            .addLayer("value", OutputLayer.Builder(LossFunctions.LossFunction.L2)
//                    .nOut(1)
//                    .activation(Activation.SOFTSIGN)
//                    .build(), "dense3")
//            .addLayer("policy", OutputLayer.Builder(LossFunctions.LossFunction.KL_DIVERGENCE)
//                    .nOut(policySize(gameSpec))
//                    .activation(Activation.SOFTMAX)
//                    .build(), "dense3")
//            .addLayer("legal", OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                    .nOut(policySize(gameSpec))
//                    .activation(Activation.SOFTSIGN)
//                    .build(), "dense3")
//            .setOutputs( "value", "policy", "legal")
//            .build()
//    val model = ComputationGraph(mconfig)
//    model.init()
//    return model
//}
//
//fun getArg(args: Array<String>, arg: String): String? {
//    for (i in args.indices) {
//        if (args[i] == "-${arg}" || args[i] == "--${arg}") return args[i+1]
//    }
//    return null
//}
//
//fun trainUsage() {
//    println("""
//        |java TrainingKt <game>
//        |        [-data <datafile>]    Will train on a file matching 'data.<datafile>.done'
//        |                              or on recent files matching data.<datafile>.nnnnn.done
//        |                              Default is <game>
//        |        [-lastn <n>]          The number of recent files to train on. Default is 10.
//        |        [-model <model>]      An optional saved model to continue training.
//        |        [-saveas <name>]      A name pattern for saved models. Default is <game>
//        |        [-rate <rate>]        Learning rate (for new models)
//        """.trimMargin())
//}
//
//fun main(args: Array<String>) {
//    if (args.contains("-h")) {
//        return trainUsage()
//    }
//    var game = args[0]
//    val filePattern = getArg(args, "data") ?: game
//    val lastN = getArg(args, "lastn")?.toInt() ?: 10
//    val modelName = getArg(args, "model")
//    val outName = getArg(args, "saveas") ?: game
//    val rate = getArg(args, "rate")?.toDouble() ?: 0.001
//
//    val gameSpec = loadSpec(game)
//    val N = gameSpec.boardSize
//    val P = gameSpec.pieceCount - 1
//    val batchSize = 300
//
//    val model = if (modelName == null) {
//        newModel(gameSpec, N, P, rate)
//    } else {
//        ModelSerializer.restoreComputationGraph(modelName)
//    }
//
//    //model.setListeners(ScoreIterationListener())
//    val uiServer = UIServer.getInstance()
//    val statsStorage = InMemoryStatsStorage()
//    uiServer.attach(statsStorage)
//    model.setListeners(StatsListener(statsStorage))
//
//    var batchCount = 0
//    val reader = FileInstanceReader(0.2, lastN, filePattern)
//    while (true) {
//        model.fit(getBatch(gameSpec, reader, batchSize))
//        batchCount++
//        if (batchCount % 1000 == 0) {
//            ModelSerializer.writeModel(model, "model.${outName}.${batchCount}", true)
//        }
//    }
//}
//
//interface InstanceReader {
//    fun next(): Instance.TrainingInstance
//}
//
//class StreamInstanceReader(val stream: InputStream) : InstanceReader {
//    override fun next(): Instance.TrainingInstance {
//        return Instance.TrainingInstance.parseDelimitedFrom(stream)
//    }
//}
//
//class FileInstanceReader(val prob: Double, val lastN: Int, val filePattern: String) : InstanceReader {
//    val rand = Random()
//
//    // return a recent file of training data matching 'data.{filepattern}.nnnnnnn.done'
//    // or else return just the one file if a full name was given instead of a pattern.
//    // we'll be constantly creating new files, and this will pick from the last N
//    fun nextStream(): FileInputStream {
//        val matcher = BiPredicate<Path, BasicFileAttributes> { file, _ ->
//            val fileName = file.fileName.toString()
//            fileName.matches(Regex(".*data.${filePattern}.[0-9]+.done")) ||
//                    fileName.toString() == filePattern
//        }
//        val paths = ArrayList<Path>()
//        for (path in find(Paths.get("."), 1, matcher).iterator()) paths.add(path)
//        val recent = paths.sortedByDescending { it.fileName }.subList(0, min(lastN, paths.size))
//        if (recent.size == 0) throw RuntimeException("No files match")
//        return FileInputStream(recent[rand.nextInt(recent.size)].toString())
//    }
//
//    var instream = nextStream()
//
//    override fun next(): Instance.TrainingInstance {
//        while (true) {
//            val instance = Instance.TrainingInstance.parseDelimitedFrom(instream)
//            if (instance == null) {
//                instream.close()
//                instream = nextStream()
//                continue
//            } else if (rand.nextDouble() < prob) {
//                return instance
//            }
//        }
//    }
//}
//
//fun parseBatch(gameSpec: GameSpec, instances: Array<Instance.TrainingInstance>): MultiDataSet {
//    val sz = gameSpec.boardSize
//    val np = gameSpec.pieceCount - 1
//    val batchSize = instances.size
//    val input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
//    val value = Nd4j.zeros(batchSize, 1)
//    val policy = Nd4j.zeros(batchSize, policySize(gameSpec))
//    val legal = Nd4j.ones(batchSize, policySize(gameSpec)).mul(-1)
//
//    for (i in 0 until batchSize) {
//        for (j in 0 until instances[i].boardState.size()) {
//            val x = j / sz
//            val y = j % sz
//            val p = instances[i].boardState.byteAt(j).toInt()
//            if (p != 0) {
//                val channel = if (p > 0) p else np - p
//                input.putScalar(intArrayOf(i, channel, x, y), 1)
//            }
//        }
//        val turn = if (instances[i].whiteMove) 1 else -1
//        for (x in 0 until sz) for (y in 0 until sz) input.putScalar(intArrayOf(i, 0, x, y), turn)
//
//        for (j in 0 until instances[i].treeSearchResultCount) {
//            val tsr = instances[i].treeSearchResultList[j]
//            policy.putScalar(intArrayOf(i, tsr.index), tsr.prob)
//            legal.putScalar(intArrayOf(i, tsr.index), 1f)
//        }
//        value.putScalar(i, instances[i].outcome)
//    }
//    return MultiDataSet(arrayOf(input), arrayOf(value, policy, legal))
//}
//
//fun getBatch(gameSpec: GameSpec, reader: InstanceReader, batchSize: Int): MultiDataSet {
//    var instances = Array(batchSize) { reader.next() }
//    return parseBatch(gameSpec, instances)
//
//}
