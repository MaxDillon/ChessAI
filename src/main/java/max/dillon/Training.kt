package max.dillon

import max.dillon.GameGrammar.GameSpec
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
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

fun main(args: Array<String>) {
    val gameSpec = loadSpec(args[0])
    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1
    val batchSize = 5000

    val mconfig = NeuralNetConfiguration.Builder()
            .learningRate(0.01)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .regularization(false)
            .graphBuilder()
            .addInputs("input")
            .addLayer("conv1", ConvolutionLayer.Builder(5, 5)
                    .nIn(2 * P + 1)
                    .stride(1, 1)
                    .padding(2, 2)
                    .nOut(30)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "input")
            .addLayer("pool", SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build(), "conv1")
            .addLayer("conv2", ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .padding(1, 1)
                    .nOut(30)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "pool")
            .addLayer("rect", DenseLayer.Builder()
                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(40)
                    .build(), "conv2")
            .addLayer("policy", OutputLayer.Builder(
                    LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .nOut(policySize(gameSpec))
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "rect")
            .addLayer("value", OutputLayer.Builder(LossFunctions.LossFunction.L2)
                    .nOut(1)
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .build(), "rect")
            .setInputTypes(InputType.convolutionalFlat(N, N, 1))
            .setOutputs("value", "policy")
            .build()
    val model = ComputationGraph(mconfig)
    model.init()

    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    var batchCount = 0
    val reader = FileInstanceReader(0.1, args[1])
    while (true) {
        model.fit(getBatch(gameSpec, reader, batchSize))
        batchCount++
        if (batchCount % 20 == 0) {
            ModelSerializer.writeModel(model, "model.${args[0]}.${batchCount}", true)
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
    var sz = gameSpec.boardSize
    var np = gameSpec.pieceCount - 1
    var batchSize = instances.size
    var input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
    var policy = Nd4j.zeros(batchSize, policySize(gameSpec))
    var value = Nd4j.zeros(batchSize, 1)

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
        var turn = if (instances[i].whiteMove) 1 else -1
        for (x in 0 until sz) for (y in 0 until sz) input.putScalar(intArrayOf(i, 0, x, y), turn)

        for (j in 0 until instances[i].treeSearchResultCount) {
            val tsr = instances[i].treeSearchResultList[j]
            policy.putScalar(tsr.index, tsr.prob)
        }
        value.putScalar(i, instances[i].outcome)
    }
    return MultiDataSet(arrayOf(input), arrayOf(value, policy))
}

fun getBatch(gameSpec: GameSpec, reader: InstanceReader, batchSize: Int): MultiDataSet {
    var instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances)
}

