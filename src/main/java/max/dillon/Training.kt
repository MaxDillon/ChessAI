package max.dillon

import com.google.protobuf.TextFormat
import max.dillon.GameGrammar.GameSpec
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
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.FileInputStream
import java.io.InputStream
import java.lang.Float
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
    val specStr = String(Files.readAllBytes(Paths.get("src/main/data/${args[0]}.textproto")))
    val builder = GameGrammar.GameSpec.newBuilder()
    TextFormat.getParser().merge(specStr, builder)
    val gameSpec = builder.apply {
        addPiece(0, builder.addPieceBuilder())
    }.build()

    val N = gameSpec.boardSize
    val P = gameSpec.pieceCount - 1

    val batchSize = 1000
    val rngSeed = 1
    val numEpochs = 200

    val random = Random()

    val config: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .learningRate(0.01)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, ConvolutionLayer.Builder(3, 3)
                    .nIn(2 * P + 1)
                    .stride(1, 1)
                    .padding(2, 2)
                    .nOut(30)
                    .activation(Activation.IDENTITY)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(2, ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .padding(3, 3)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build())
            .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(4, DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nOut(500)
                    .build())
            .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                    .nOut(policySize(gameSpec))
                    .activation(Activation.SOFTMAX)
                    .build())
            .setInputType(InputType.convolutionalFlat(N, N, 1)) //See note below
            .backprop(true)
            .pretrain(false)
            .build();

    val model = MultiLayerNetwork(config)
    model.init()
    model.setListeners(ScoreIterationListener(5))

    val instream = FileInputStream(args[1])
    for (i in 1..numEpochs) {
        val (input, output) = getBatch(gameSpec, instream, batchSize)
        model.fit(input, output)
    }
}

fun getBatch(gameSpec: GameSpec, instream: InputStream, batchSize: Int): Pair<INDArray, INDArray> {
    var sz = gameSpec.boardSize
    var np = gameSpec.pieceCount - 1
    var input = Nd4j.zeros(batchSize, 2 * np + 1, sz, sz)
    var output = Nd4j.zeros(batchSize, policySize(gameSpec))

    for (i in 0 until batchSize) {
        val inst = Instance.TrainingInstance.parseDelimitedFrom(instream)
        if (inst == null) {
            println("Out of data")
            break
        }
        for (i in 0 until inst.boardState.size()) {
            val x = i / sz
            val y = i % sz
            val p = inst.boardState.byteAt(i).toInt()
            if (p != 0) {
                val channel = if (p > 0) p else np - p
                input.putScalar(intArrayOf(i, channel, x, y), 1)
            }
        }
        var turn = if (inst.whiteMove) 0 else 1
        for (x in 0 until sz) for (y in 0 until sz) input.putScalar(intArrayOf(i, 0, x, y), turn)

        for (i in 0 until inst.treeSearchResultCount) {
            val tsr = inst.treeSearchResultList[i]
            output.putScalar(tsr.index, if (Float.isNaN(tsr.meanValue)) 0f else tsr.meanValue) // think this should actually be pi
        }
    }
    return Pair(input, output)
}
