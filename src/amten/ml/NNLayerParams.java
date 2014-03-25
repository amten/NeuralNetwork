package amten.ml;

import java.io.Serializable;

/**
 * Parameters for a layer in a neural network.
 * Layer can be fully connected or convolutional.
 *
 * @author Johannes Amtén
 */
public class NNLayerParams implements Serializable {
    public int numFeatures = 0;
    public int patchWidth = 0;
    public int patchHeight = 0;
    public int poolWidth = 0;
    public int poolHeight = 0;

    /**
     * Create a definition of a fully connected layer.
     */
    public NNLayerParams(int numFeatures) {
        this.numFeatures = numFeatures;
    }

    /**
     * Create a definition of a convolutional connected layer.
     */
    public NNLayerParams(int numFeatures, int patchWidth, int patchHeight, int poolWidth, int poolHeight) {
        this.numFeatures = numFeatures;
        this.patchWidth = patchWidth;
        this.patchHeight = patchHeight;
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
    }

    public boolean isConvolutional() {
        return patchWidth > 0 && patchHeight > 0;
    }

    public boolean isPooled() {
        return poolWidth > 1 && poolHeight > 1;
    }
}
