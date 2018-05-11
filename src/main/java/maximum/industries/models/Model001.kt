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
 * Two residual blocks. Policy and Legal off of last residual. Value off of short
 * dense tower.
 *
 * ./consistency.sh chess model.chess.Model001.17000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.424	0.576	0.010	0.091	0.056	0.009
 *	BLACK	0.415	0.585	0.013	0.071	0.049	0.008
 */
class Model001 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec): ComputationGraph {
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 48

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
                    convolution(inChannels, convFilters)
                }

                // ######## Residual block    #############################
                .R("res1", "conv1") {
                    convolution(filters = convFilters)
                    convolution(filters = convFilters)
                }
                .R("res2", "res1") {
                    convolution(filters = convFilters)
                    convolution(filters = convFilters)
                }

                // ######## Value head        #############################
                .F("valuetower", "res2") {
                    dense(30)
                    dense(10)
                }
                .F("value", "valuetower") {
                    output(nOut = 1,
                           activation = Activation.SOFTSIGN,
                           loss = LossFunctions.LossFunction.L2)
                }
                // ######## Policy head      #############################
                .F("policy1", "res2") {
                    convolution(filters = policyChannels(gameSpec))
                }
                .CNN2FF("policyff", "policy1", sz, policyChannels)
                .F("policy", "policyff") {
                    loss(activation = Activation.SOFTMAX,
                         loss = LossFunctions.LossFunction.KL_DIVERGENCE)
                }
                // ######## Legal head       #############################
                .F("legal1", "res2") {
                    convolution(filters = policyChannels(gameSpec))
                }
                .CNN2FF("legalff", "legal1", sz, policyChannels)
                .F("legal", "legalff") {
                    loss(activation = Activation.SOFTSIGN,
                         loss = LossFunctions.LossFunction.MSE)
                }

                .setOutputs("value", "policy", "legal")
                .build()

        return ComputationGraph(config).apply { init() }
    }
}

