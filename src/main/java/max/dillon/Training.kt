package max.dillon

import com.google.protobuf.TextFormat
import max.dillon.GameGrammar.GameSpec
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.FileInputStream
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Paths
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

    val config: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .learningRate(0.01)
            .regularization(false).l2(1e-5)
            .list()
            .layer(0, ConvolutionLayer.Builder(3, 3)
                    .nIn(2 * P + 1)
                    .stride(1, 1)
                    .padding(1, 1)
                    .nOut(15)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build())
//            .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                    .kernelSize(2, 2)
//                    .stride(2, 2)
//                    .build())
//            .layer(1, ConvolutionLayer.Builder(3, 3)
//                    .stride(1, 1)
//                    .padding(1, 1)
//                    .nOut(25)
//                    .activation(Activation.IDENTITY)
//                    .weightInit(WeightInit.XAVIER)
//                    .build())
//            .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                    .kernelSize(2, 2)
//                    .stride(2, 2)
//                    .build())
            .layer(1, DenseLayer.Builder()
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(20)
                    .build())
//            .layer(2, DenseLayer.Builder()
//                    .activation(Activation.ELU)
//                    .weightInit(WeightInit.XAVIER)
//                    .nOut(20)
//                    .build())
            .layer(2, OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .nOut(policySize(gameSpec))
                    .activation(Activation.SOFTMAX)
                    .build())
            .setInputType(InputType.convolutionalFlat(N, N, 1)) //See note below
            .backprop(true)
            .pretrain(false)
            .build()

    val model = MultiLayerNetwork(config)
    model.init()
    model.setListeners(ScoreIterationListener(1))

    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    model.setListeners(StatsListener(statsStorage))

    var batchCount = 0
    val reader = FileInstanceReader(0.1, args[1])
    while (true) {

        val (input, output) = getBatch(gameSpec, reader, batchSize)
        batchCount++
        model.fit(input, output)

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

fun parseBatch(gameSpec: GameSpec,  instances: Array<Instance.TrainingInstance>): Pair<INDArray,INDArray> {
    var sz = gameSpec.boardSize
    var np = gameSpec.pieceCount - 1
    var batchSize = instances.size
    var input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
    var output = Nd4j.zeros(batchSize, policySize(gameSpec))

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
            output.putScalar(tsr.index, tsr.prob)
        }
    }
    return Pair(input,output)
}

fun getBatch(gameSpec: GameSpec, reader: InstanceReader, batchSize: Int): Pair<INDArray, INDArray> {
    var instances = Array(batchSize) { reader.next() }
    return parseBatch(gameSpec, instances)
}
