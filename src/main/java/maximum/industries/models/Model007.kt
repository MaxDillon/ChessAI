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
 * TANH for value. Same as Model006 but using momentum instead of ADAM. Best
 * results so far!!
 *
 * ./consistency.sh chess model.chess.Model007.13000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.801	0.199	-0.029	0.000	0.000	0.008
 *	BLACK	0.834	0.166	-0.061	0.000	0.000	0.009   << probably an anomaly
 *
 * ./consistency.sh chess model.chess.Model007.18000 20
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.828	0.172	-0.009	0.000	0.000	0.009
 *	BLACK	0.829	0.171	-0.018	0.000	0.000	0.009
 *
 * After 20k trained with very low drawweight so we mostly have wins/losses.
 * After 28k training on more recent data with exploration=0.3 to emphasize the
 * good moves more. Previously exploration was so high that dists were flat.
 * Hard to find correlation there.
 *
 * ./consistency.sh chess model.chess.Model007.63000 -mindepth 20 -maxdepth 50 -maxbatch 60 -states 500
 *	Player	+Mass	-Mass	Correl	+Pos	-Pos	Frac+
 *	WHITE	0.905	0.095	-0.052	0.000	0.000	0.008
 *	BLACK	0.903	0.097	-0.015	0.000	0.000	0.008

 *
 */
class Model007 : IModel {
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

