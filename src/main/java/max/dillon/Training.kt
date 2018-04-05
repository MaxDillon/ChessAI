package max.dillon

import max.dillon.GameGrammar.GameSpec
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.FileInputStream
import java.io.InputStream
import java.util.*

fun policySize(gameSpec: GameSpec): Int {
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    val src_dim = (
            if (gameSpec.moveSource == GameGrammar.MoveSource.ENDS) P
            else N * N)
    val dst_dim = N * N
    return src_dim * dst_dim
}

fun newModel(gameSpec: GameSpec, N: Int, P: Int): ComputationGraph {
    val mconfig = NeuralNetConfiguration.Builder()
            .learningRate(0.005)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .regularization(true).l2(1e-6)
            .graphBuilder()
            .addInputs("input")
            .setInputTypes(InputType.convolutional(N, N,3))
            .addLayer("conv1", ConvolutionLayer.Builder(3, 3)
                    .nIn(2 * P + 1)
                    .stride(1, 1)
                    .padding(1, 1)
                    .nOut(40)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "input")
            .addLayer("batch1", BatchNormalization(),
                      "conv1")
            .addLayer("rect1", ActivationLayer.Builder()
                    .activation(Activation.RELU)
                    .build(), "batch1")
            .addLayer("conv2", ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .padding(1, 1)
                    .nOut(40)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "rect1")
            .addLayer("batch2", BatchNormalization(),
                      "conv2")
            .addLayer("rect2", ActivationLayer.Builder()
                    .activation(Activation.RELU)
                    .build(), "batch2", "input")
            .addLayer("dense1", DenseLayer.Builder()
                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(50)
                    .build(), "rect2")
            .addLayer("dense2", DenseLayer.Builder()
                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(30)
                    .build(), "dense1")
            .addLayer("value", OutputLayer.Builder(LossFunctions.LossFunction.L2)
                    .nOut(1)
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "dense2")
            .addLayer("policy", OutputLayer.Builder(LossFunctions.LossFunction.KL_DIVERGENCE)
                    .nOut(policySize(gameSpec))
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "dense2")
            .addLayer("legal", OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nOut(policySize(gameSpec))
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "dense2")
            .setOutputs( "value", "policy", "legal")
            .build()
    val model = ComputationGraph(mconfig)
    model.init()
    return model
}

fun main(args: Array<String>) {
    var baseName = args[0]
    val gameSpec = loadSpec(baseName)
    val dataFile = args[1]
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    val batchSize = 1500

    val model = if (args.size < 3) {
        newModel(gameSpec, N, P)
    } else {
        baseName = args[2]
        ModelSerializer.restoreComputationGraph(args[2])
    }

    //model.setListeners(ScoreIterationListener())
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    var batchCount = 0
    val reader = FileInstanceReader(0.1, dataFile)
    while (true) {
        model.fit(getBatch(gameSpec, reader, batchSize))
        batchCount++
        if (batchCount % 1000 == 0) {
            ModelSerializer.writeModel(model, "model.${baseName}.${batchCount}", true)
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

class FileInstanceReader(val prob: Double, val file: String) : InstanceReader {
    val rand = Random()
    var instream = FileInputStream(file)

    override fun next(): Instance.TrainingInstance {
        while (true) {
            val instance = Instance.TrainingInstance.parseDelimitedFrom(instream)
            if (instance == null) {
                instream.close()
                instream = FileInputStream(file)
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
        val turn = if (instances[i].whiteMove) 1 else -1
        for (x in 0 until sz) for (y in 0 until sz) input.putScalar(intArrayOf(i, 0, x, y), turn)

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
    var instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances)
}

