package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Approximately the original quest model for connect4. Three convolutional
 * layers with skip connections (input=>conv2, conv1=>conv3). Not added as in
 * a residual block though.
 *
 * Followed by a dense tower. Value, Policy, Legal off of dense tower.
 *
 * Doesn't seem surprising that this model would have trouble with chess. Seems
 * unlikely we can reconstruct such a large policy after passing through just
 * 30 hidden units. After 6000 batches the score still has slope, but the stats
 * show we're finding concepts very slowly.
 *
 * ./consistency.sh chess model.chess.Model000.8000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.064	0.936	0.055	0.503	0.495	0.008
 *	BLACK	0.083	0.917	0.019	0.491	0.492	0.009
 *
 * Note: update parameter ratios are super small for policy and legal (esp.
 * legal layers). Why? Maybe just the size question again.
 */
class Model000 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec): ComputationGraph {
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val convFilters = 64

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).l2(1e-7)
                .updater(Updater.NESTEROVS)
                .learningRate(0.001)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(sz, sz, inChannels))

                // ######## First convolution #############################
                .F("conv1", "input") {
                    convolution(inChannels, convFilters, norm=false)
                }
                .F("conv2", "conv1", "input") {
                    convolution(filters = convFilters)
                }
                .F("conv3", "conv2", "conv1") {
                    convolution(filters = convFilters)
                }
                .F("dense", "conv3") {
                    dense(50)
                    dense(40)
                    dense(30)
                }

                // ######## Value head        #############################
                .F("value", "dense") {
                    output(nOut = 1,
                           activation = Activation.SOFTSIGN,
                           loss = LossFunctions.LossFunction.L2)
                }
                // ######## Policy head      #############################
                .F("policy", "dense") {
                    output(nOut = policySize(gameSpec),
                           activation = Activation.SOFTMAX,
                           loss = LossFunctions.LossFunction.KL_DIVERGENCE)
                }
                // ######## Legal head       #############################
                .F("legal", "dense") {
                    output(nOut = policySize(gameSpec),
                           activation = Activation.SOFTSIGN,
                           loss = LossFunctions.LossFunction.MSE)
                }
                .setOutputs("value", "policy", "legal")
                .build()

        return ComputationGraph(config).apply { init() }
    }
}

