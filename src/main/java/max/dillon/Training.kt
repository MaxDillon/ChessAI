package max.dillon

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

import org.deeplearning4j.base.MnistFetcher
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

class MyMnistFetcher() : MnistFetcher() {
    override fun getTrainingFilesURL(): String {
        return "http://deeplearning4j-resources.westus2.cloudapp.azure.com/mnist/train-images-idx3-ubyte.gz"
    }
    override fun getTrainingFileLabelsURL(): String {
        return "http://deeplearning4j-resources.westus2.cloudapp.azure.com/mnist/train-labels-idx1-ubyte.gz"
    }
    override fun getTestFilesURL(): String {
        return "http://deeplearning4j-resources.westus2.cloudapp.azure.com/mnist/t10k-images-idx3-ubyte.gz"
    }
    override fun getTestFileLabelsURL(): String {
        return "http://deeplearning4j-resources.westus2.cloudapp.azure.com/mnist/t10k-labels-idx1-ubyte.gz"
    }
}

fun main(args: Array<String>) {
    val batchSize = 128
    val rngSeed = 1
    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    var numEpochs = 20

    // HACK: the MNIST hosting location has changed and the fetcher we have in the dl4j examples
    // points to the old location. Further, it's hard-coded in the fetcher. We have to subclass and
    // override the fetcher's location accessors and then trigger the download ourselves so that the
    // files will be cached locally in the expected location.
    val fetcher = MyMnistFetcher()
    fetcher.downloadAndUntar()

    val mat = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), intArrayOf(2, 3))

    val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

    val config: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .learningRate(0.05)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, DenseLayer.Builder()
                    .nIn(numRows * numColumns)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false)
            .backprop(true)
            .build();


    val model = MultiLayerNetwork(config)
    model.init()
    model.setListeners(ScoreIterationListener(1))

    for (i in 1..numEpochs) {
        model.fit(mnistTrain)
    }
}


//private val log = LoggerFactory.getLogger(Test::class.java!!)
//
//@JvmStatic
//fun main(args: Array<String>) {
//
//    try {
//        val boardSize = 8
//        val numPieceTypes = 6
//        val player = 0
//        val piece = 1
//        val row = 0
//        val col = 0
//
//        // construct an INDArray from a regular java array
//        val mat = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), intArrayOf(2, 3))
//
//        // construct an INDArray of zeroes with the given shape and set individual values
//        val state = Nd4j.zeros(2, numPieceTypes, boardSize, boardSize)
//        state.putScalar(intArrayOf(player, piece, row, col), 1.0)
//
//
//        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
//        val labelIndex = 4     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//        val numClasses = 3     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2
//
//        val batchSizeTraining = 30    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
//        val trainingData = readCSVDataset(
//                "/DataExamples/animals/animals_train.csv",
//                batchSizeTraining, labelIndex, numClasses)
//
//
//        // this is the data we want to classify
//        val batchSizeTest = 44
//        val testData = readCSVDataset("/DataExamples/animals/animals.csv",
//                                      batchSizeTest, labelIndex, numClasses)
//
//
//        // make the data model for records prior to normalization, because it
//        // changes the data.
//        val animals = makeAnimalsForTesting(testData)
//
//
//        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
//        val normalizer = NormalizerStandardize()
//        normalizer.fit(trainingData)           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        normalizer.transform(trainingData)     //Apply normalization to the training data
//        normalizer.transform(testData)         //Apply normalization to the test data. This is using statistics calculated from the *training* set
//
//        val numInputs = 4
//        val outputNum = 3
//        val iterations = 1000
//        val seed: Long = 6
//
//        log.info("Build model....")
//        val conf = NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(iterations)
//                .activation(Activation.TANH)
//                .weightInit(WeightInit.XAVIER)
//                .learningRate(0.1)
//                .regularization(true).l2(1e-4)
//                .list()
//                .layer(0, DenseLayer.Builder().nIn(numInputs).nOut(3).build())
//                .layer(1, DenseLayer.Builder().nIn(3).nOut(3).build())
//                .layer(2, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
//                .backprop(true).pretrain(false)
//                .build()
//
//        //run the model
//        val model = MultiLayerNetwork(conf)
//        model.init()
//        model.setListeners(ScoreIterationListener(100))
//
//        model.fit(trainingData)
//
//        //evaluate the model on the test set
//        val eval = Evaluation(3)
//        val output = model.output(testData.getFeatureMatrix())
//
//        eval.eval(testData.getLabels(), output)
//        log.info(eval.stats())
//
//        setFittedClassifiers(output, animals)
//        logAnimals(animals)
//
//    } catch (e: Exception) {
//        e.printStackTrace()
//    }
//
//}
//
