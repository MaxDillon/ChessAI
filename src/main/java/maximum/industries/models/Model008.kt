package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Four residual blocks after first convolution. Policy off of last residual.
 * Value off of shorter dense tower. No legal. More convolutional filters.
 * TANH for value. Same as Model007, but using Adam.  But the MAIN thing is
 * that the KL_DIVERGENCE was an incorrect thing to use for policy. This did
 * not optimize all and that component of the loss was stuck.  MCXENT trains.
 * Also using ELU instead of RELU since some say that's the new hotness and
 * XAVIER since Karpathy says so even with RELU apparently. Though with all
 * the batch norm this shouldn't matter too much.
 *
 * ./consistency.sh chess model.chess.Model008.2000     !! sweet start
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.645	0.355	-0.025	0.000	0.000	0.009
 *	BLACK	0.634	0.366	-0.071	0.000	0.000	0.009
 *
 *  ./consistency.sh chess model.chess.Model008.15000
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.775	0.225	-0.053	0.000	0.000	0.009
 *	BLACK	0.810	0.190	-0.010	0.000	0.000	0.009
 *
 * Likely overfitting ... running with new regularization and learning rate.
 *
 */
class Model008 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec,
                          learningRateOverride: Double?,
                          regularizationOverride: Double?): ComputationGraph {
        val learningRate = learningRateOverride ?: 5e-3
        val regularization = regularizationOverride ?: 0.25

        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 128

        val activation = Activation.ELU
        val weightInit = WeightInit.XAVIER

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(regularization)
                .updater(Nesterovs(learningRate))
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
                .F("valuetower", "res4") {
                    dense(20, init = weightInit, activation = activation)
                }
                .F("value", "valuetower") {
                    output(nOut = 1,
                           activation = Activation.TANH,
                           loss = LossFunctions.LossFunction.L2)
                }
                // ######## Policy head      #############################
                .F("policy1", "res4") {
                    convolution(filters = policyChannels(gameSpec), init = weightInit, activation = activation)
                }
                .CNN2FF("policyff", "policy1", sz, policyChannels)
                .F("policy", "policyff") {
                    loss(activation = Activation.SOFTMAX,
                         loss = LossFunctions.LossFunction.KL_DIVERGENCE)
                }

                .setOutputs("value", "policy")
                .build()

        return ComputationGraph(config)
    }
}

