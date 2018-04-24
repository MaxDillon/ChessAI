package maximum.industries.models

import maximum.industries.*
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * Three residual blocks after first convolution. Policy off of last residual.
 * Value off of shorter dense tower. No legal. More convolutional filters.
 * TANH for value. ADAM
 *
 * ./consistency.sh chess model.chess.Model006.6000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.601	0.399	0.003	0.000	0.000	0.008
 *	BLACK	0.572	0.428	0.006	0.000	0.000	0.009
 *
 * ./consistency.sh chess model.chess.Model006.17000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.718	0.282	0.001	0.000	0.000	0.009
 *	BLACK	0.740	0.260	-0.019	0.000	0.000	0.008
 *
 * ./consistency.sh chess model.chess.Model006.30000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.773	0.227	-0.015	0.000	0.000	0.008
 *	BLACK	0.805	0.195	0.002	0.000	0.000	0.008
 *
 * ./consistency.sh chess model.chess.Model006.35000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.795	0.205	-0.016	0.000	0.000	0.009
 *	BLACK	0.808	0.192	-0.028	0.000	0.000	0.008
 *
 *./consistency.sh chess model.chess.Model006.55000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.839	0.161	-0.002	0.000	0.000	0.008
 *	BLACK	0.837	0.163	-0.034	0.000	0.000	0.009
 *
 * Note: after model 35000 changed the trainer to present fewer draws on
 * the theory that we'll learn useful things faster from wins and losses.
 *
 * Note: after model 41000 switched to momentum updater because we were
 * having NaN crashes or other excursions with ADAM. Not sure if this will
 * help ... ACTUALLY .... IT DOESN'T LOOK LIKE THIS METHOD OF TRANSFERRING
 * PARAMETERS WORKS AT ALL.
 *
 */
class Model006 : IModel {
    override fun newModel(gameSpec: GameGrammar.GameSpec): ComputationGraph {
        val sz = gameSpec.boardSize
        val inChannels = inputChannels(gameSpec)
        val policyChannels = policyChannels(gameSpec)
        val convFilters = 128

        val config = NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).l2(1e-6)
                .updater(Updater.NESTEROVS)
                .learningRate(5e-3)

//             Used thru 41000
//                .updater(Updater.ADAM)
//                .learningRate(0.01)

//                .updater(Adam.builder().learningRate(0.005)
//                                 .beta1(0.9).beta2(0.999)
//                                 .epsilon(8e-8).build())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(sz, sz, inChannels))

                // ######## First convolution #############################
                .F("conv1", "input") {
                    convolution(inChannels, convFilters, norm=false)
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
                .R("res3", "res2") {
                    convolution(filters = convFilters)
                    convolution(filters = convFilters)
                }

                // ######## Value head        #############################
                .F("valuetower", "res3") {
                    dense(20)
                }
                .F("value", "valuetower") {
                    output(nOut = 1,
                           activation = Activation.TANH,
                           loss = LossFunctions.LossFunction.L2)
                }
                // ######## Policy head      #############################
                .F("policy1", "res3") {
                    convolution(filters = policyChannels(gameSpec))
                }
                .CNN2FF("policyff", "policy1", sz, policyChannels)
                .F("policy", "policyff") {
                    loss(activation = Activation.SOFTMAX,
                         loss = LossFunctions.LossFunction.KL_DIVERGENCE)
                }

                .setOutputs("value", "policy")
                .build()

        return ComputationGraph(config).apply { init() }
    }
}

