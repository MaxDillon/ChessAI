package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Two residual blocks. Policy and legal off of last residual. Value off of shorter
 * dense tower. More convolutional filters than Model001.
 *
 * ./consistency.sh chess model.chess.Model004.17000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.497	0.503	0.031	0.062	0.041	0.009
 *	BLACK	0.480	0.520	-0.020	0.072	0.044	0.008
 */
class Model004 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec,
                          learningRateOverride: Double?,
                          regularizationOverride: Double?): ComputationGraph {
        val learningRate = learningRateOverride ?: 0.001
        val regularization = regularizationOverride ?: 1e-7
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 80

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(regularization)
                .updater(Nesterovs(learningRate))
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
                    dense(20)
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

        return ComputationGraph(config)
    }
}

