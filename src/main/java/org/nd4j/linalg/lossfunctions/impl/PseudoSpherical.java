package org.nd4j.linalg.lossfunctions.impl;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class PseudoSpherical implements ILossFunction {
    private static final double DEFAULT_SOFTMAX_CLIPPING_EPSILON = 1e-8;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray weights;

    private double softmaxClipEps;

    public PseudoSpherical() {
        this(null);
    }

    public PseudoSpherical(INDArray weights) {
        this(DEFAULT_SOFTMAX_CLIPPING_EPSILON, weights);
    }

    public PseudoSpherical(@JsonProperty("softmaxClipEps") double softmaxClipEps, @JsonProperty("weights") INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        if (softmaxClipEps < 0 || softmaxClipEps > 0.5) {
            throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: "
                    + softmaxClipEps);
        }
        this.weights = weights;
        this.softmaxClipEps = softmaxClipEps;
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1)
                    + ") does not match output layer" + " number of outputs (nOut = " + preOutput.size(1)
                    + ") ");

        }
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        if (activationFn instanceof ActivationSoftmax && softmaxClipEps > 0.0) {
            BooleanIndexing.replaceWhere(output, softmaxClipEps, Conditions.lessThan(softmaxClipEps));
            BooleanIndexing.replaceWhere(output, 1.0 - softmaxClipEps, Conditions.greaterThan(1.0 - softmaxClipEps));
        }

        INDArray numerators = Transforms.pow(output, 0.5).sum(1);
        INDArray scoreArr = Transforms.pow(output, -0.5).muliColumnVector(numerators).muli(labels);

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != scoreArr.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length()
                        + ") does not match output.size(1)=" + preOutput.size(1));
            }
            scoreArr.muliRowVector(weights);
        }

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                               boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        double score = scoreArr.sumNumber().doubleValue();
        if (average) {
            score /= scoreArr.size(0);
        }
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1)
                    + ") does not match output layer" + " number of outputs (nOut = " + preOutput.size(1)
                    + ") ");

        }
        long[] shape = preOutput.shape();
        long batchSize = shape[0];
        long nOut = shape[1];

        INDArray sj = preOutput
                .reshape(batchSize * nOut, 1)
                .broadcast(batchSize * nOut, nOut)
                .reshape(batchSize, nOut, nOut);
        INDArray si = sj.permute(0, 2, 1); // creates a view; does not duplicate
        INDArray exp = Transforms.exp(sj.sub(si).divi(2)); // transforms are in-place; not sure subi would be safe
        INDArray t = labels
                .reshape(batchSize * nOut, 1)
                .broadcast(batchSize * nOut, nOut)
                .reshape(batchSize, nOut, nOut)
                .permute(0, 2, 1);
        INDArray g = t.muli(exp).divi(2);
        INDArray grad = g.sum(2).subi(g.sum(1));

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != preOutput.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length()
                        + ") does not match output.size(1)=" + preOutput.size(1));
            }
            grad.muliRowVector(weights);
        }

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }

        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                                                          INDArray mask, boolean average) {
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String name() {
        return toString();
    }


    @Override
    public String toString() {
        if (weights == null)
            return "PseudoSpherical()";
        return "PseudoSpherical(weights=" + weights + ")";
    }
}