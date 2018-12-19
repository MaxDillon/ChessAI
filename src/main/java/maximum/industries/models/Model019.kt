package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Same as Model011 but with an extra dense layer in the value head, and one more residual block,
 * and more convolutional filters, and a higher learning rate.
 */
class Model019 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec,
                          learningRateOverride: Double?,
                          regularizationOverride: Double?): ComputationGraph {
        val learningRate = learningRateOverride ?: 5e-2
        val regularization = regularizationOverride ?: 1e-3

        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 128

        val activation = Activation.RELU
        val weightInit = WeightInit.RELU

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(regularization)
                .updater(Adam(learningRate))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(sz.toLong(), sz.toLong(), inChannels.toLong()))

                // ######## First convolution #############################
                .F("conv1", "input") {
                    convolution(inChannels, convFilters, kernel = 5, pad = 2, init = WeightInit.XAVIER, activation = activation,
                                norm = false)
                }

                // ######## Residual block    #############################
                .R("res1", "conv1") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }
                .R("res2", "res1") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }
                .R("res3", "res2") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }
                .R("res4", "res3") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }
                .R("res5", "res4") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }
                .R("res6", "res5") {
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                    convolution(filters = convFilters, init = weightInit, activation = activation)
                }

                // ######## Value head        #############################
                .F("valuetower", "res6") {
                    dense(40, init = weightInit, activation = activation)
                    dense(20, init = weightInit, activation = activation)
                }
                .F("value", "valuetower") {
                    output(nOut = 1,
                           activation = Activation.TANH,
                           loss = LossFunctions.LossFunction.L2,
                           norm = false)
                }
                // ######## Policy head      #############################
                .F("policytower", "res6") {
                    dense(100, init = weightInit, activation = activation)
                    dense(policySize(gameSpec), init = weightInit, activation = activation)
                }
                .F("policy", "policytower") {
                    loss(activation = Activation.SOFTMAX,
                         loss = LossFunctions.LossFunction.MCXENT)
                }

                .setOutputs("value", "policy")
                .build()

        return ComputationGraph(config)
    }
}

