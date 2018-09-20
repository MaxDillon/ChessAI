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
 * One residual block followed by a large dense tower. Policy and Value off of dense
 * layers. Legal off of residual block.
 *
 * Not yet run. Don't believe suitable for chess.
 */
class Model002 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec,
                          learningRateOverride: Double?,
                          regularizationOverride: Double?): ComputationGraph {
        val learningRate = learningRateOverride ?: 0.001
        val regularization = regularizationOverride ?: 1e-7
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 64

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(regularization)
                .updater(Nesterovs(learningRate))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(sz.toLong(), sz.toLong(), inChannels.toLong()))

                // ######## First convolution #############################
                .F("conv1", "input") {
                    convolution(inChannels, convFilters, norm=false)
                }

                // ######## Residual block    #############################
                .R("res1", "conv1") {
                    convolution(filters = convFilters)
                    convolution(filters = convFilters)
                }
                .F("dense", "res1") {
                    dense(128)
                    dense(64)
                    dense(32)
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
                .F("legal1", "res1") {
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

