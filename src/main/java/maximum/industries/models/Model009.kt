package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Attempt to use newly defined pseudo-spherical loss
 */
class Model009 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec): ComputationGraph {
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 128

        val activation = Activation.ELU
        val weightInit = WeightInit.XAVIER

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).l2(0.1)
                .updater(Updater.NESTEROVS)
                .learningRate(1e-7)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(sz, sz, inChannels))

                // ######## First convolution #############################
                .F("conv1", "input") {
                    convolution(inChannels, convFilters, init = WeightInit.XAVIER, activation = activation,
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

                // ######## Value head        #############################
//                .F("valuetower", "res4") {
//                    dense(20, init = weightInit, activation = activation)
//                }
//                .F("value", "valuetower") {
//                    output(nOut = 1,
//                           activation = Activation.TANH,
//                           loss = LossFunctions.LossFunction.L2)
//                }
                // ######## Policy head      #############################
                .F("policy1", "res4") {
                    convolution(filters = policyChannels(gameSpec), init = weightInit, activation = activation)
                }
                .CNN2FF("policyff", "policy1", sz, policyChannels)
                .F("policy", "policyff") {
                    loss(activation = Activation.SOFTMAX,
                         loss = LossFunctions.LossFunction.PSEUDO_SPHERICAL)
                }

                .setOutputs("policy")
                .build()

        return ComputationGraph(config).apply { init() }
    }
}

