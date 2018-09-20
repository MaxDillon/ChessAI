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
 * Two residual blocks. Policy off of last residual. Value off of shorter
 * dense tower. No legal. More convolutional filters than Model001.
 *
 * These runs suggest that the legal output is unnecessary and maybe harmful.
 * We seem to concentrate probability mass on legal moves just as well without
 * it. And we seem to learn the legal output more slowly in configs where
 * we have it. Note that for chess here we haven't learned much correlation yet.
 *
 *  ./consistency.sh chess model.chess.Model005.17000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.484	0.516	0.005	0.000	0.000	0.009
 *	BLACK	0.509	0.491	-0.007	0.000	0.000	0.008
 *
 *  ./consistency.sh chess model.chess.Model005.86000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.704	0.296	0.008	0.000	0.000	0.009
 *	BLACK	0.689	0.311	-0.014	0.000	0.000	0.009

 */
class Model005 : IModel {
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
                .setInputTypes(InputType.convolutional(sz.toLong(), sz.toLong(), inChannels.toLong()))

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

                .setOutputs("value", "policy")
                .build()

        return ComputationGraph(config)
    }
}

