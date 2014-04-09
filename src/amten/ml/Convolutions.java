package amten.ml;

import amten.ml.matrix.Matrix;

/**
 * Helper class with calculations and matrix transformations pertaining to convolutions.
 *
 * @author Johannes Amtén
 */
public class Convolutions {

    public static PoolingResult maxPool(Matrix inputs, int inputWidth, int inputHeight, int poolWidth, int poolHeight) {
        // Input pixel coordinates are stored as rows in m.
        // Each column in m is for a different channel.
        // (Channels for input layer contain R/G/B-values if colour-image.)
        // (Channels for hidden layer contains the features of the previous layer.)
        int inputSize = inputHeight*inputWidth;
        int numExamples = inputs.numRows() / inputSize;
        int numChannels = inputs.numColumns();
        int outputHeight = inputHeight % poolHeight == 0 ?  inputHeight / poolHeight :
                                                            inputHeight / poolHeight +1 ;
        int outputWidth = inputWidth % poolWidth == 0 ? inputWidth / poolWidth :
                                                        inputWidth / poolWidth + 1;
        int outputSize = outputHeight*outputWidth;
        Matrix outputs = new Matrix(numExamples*outputSize, numChannels);
        Matrix prePoolRowIndexes = new Matrix(outputs.numRows(), outputs.numColumns());
        for (int example = 0; example < numExamples; example++) {
            for (int outputY = 0; outputY < outputHeight; outputY++) {
                for (int outputX = 0; outputX < outputWidth; outputX++) {
                    for (int channel = 0; channel < numChannels; channel++) {
                        double maxValue = Double.NEGATIVE_INFINITY;
                        int maxInputsRowIndex = 0;
                        for (int inputY = outputY*poolHeight; inputY < outputY*poolHeight+poolHeight && inputY < inputHeight; inputY++) {
                            for (int inputX = outputX*poolWidth; inputX < outputX*poolWidth+poolWidth && inputX < inputWidth; inputX++) {
                                int inputsRowIndex = example*inputSize + inputY*inputWidth + inputX;
                                double value = inputs.get(inputsRowIndex, channel);
                                if (value > maxValue) {
                                    maxValue = value;
                                    maxInputsRowIndex = inputsRowIndex;
                                }
                            }
                        }
                        int outputsRowIndex = example*outputSize + outputY*outputWidth + outputX;
                        outputs.set(outputsRowIndex, channel, maxValue);
                        if (prePoolRowIndexes != null) {
                            prePoolRowIndexes.set(outputsRowIndex, channel, maxInputsRowIndex);
                        }
                    }
                }
            }
        }
        return new PoolingResult(outputs, prePoolRowIndexes);
    }

    public static Matrix antiPoolDelta(Matrix delta, Matrix prePoolRowIndexes, int numRowsPrePool) {
        Matrix result = new Matrix(numRowsPrePool, delta.numColumns());
        for (int row = 0; row < delta.numRows(); row++) {
            for (int col = 0; col < delta.numColumns(); col++) {
                result.set((int) prePoolRowIndexes.get(row, col), col, delta.get(row, col));
            }
        }
        return result;
    }

    public static Matrix generatePatchesFromInputLayer(Matrix inputs, int inputWidth, int inputHeight, int patchWidth, int patchHeight) {
        // Input data has one row per example.
        // Input data has one column per pixel.
        //      Assumes pixel-numbers are generated row-wise.
        //      (i.e. first all columns of row 0, then all columns of row 1 etc.)
        //      This is how the image-data is represented in the Kaggle Digits competition. (http://www.kaggle.com/c/digit-recognizer/data)
        // Output data have one row per example/patch
        // Output data have one column per patchPixel.

        int numChannels = inputs.numColumns() / (inputWidth*inputHeight);
        int numPatchesPerExample = (inputWidth-patchWidth+1)*(inputHeight-patchHeight+1);
        int numExamples = inputs.numRows();
        Matrix output = new Matrix(numExamples*numPatchesPerExample, numChannels*patchWidth*patchHeight);
        for (int example = 0; example < numExamples; example++) {
            int patchNum = 0;
            for (int inputStartY = 0; inputStartY < inputHeight-patchHeight+1; inputStartY++) {
                for (int inputStartX = 0; inputStartX < inputWidth-patchWidth+1; inputStartX++) {
                    for (int channel = 0; channel < numChannels; channel++) {
                        for (int patchPixelY = 0; patchPixelY < patchHeight; patchPixelY++) {
                            for (int patchPixelX = 0; patchPixelX < patchWidth; patchPixelX++) {
                                int inputY = inputStartY + patchPixelY;
                                int inputX = inputStartX + patchPixelX;
                                double value = inputs.get(example, channel*inputHeight*inputWidth + inputY*inputWidth + inputX);
                                output.set(example*numPatchesPerExample + patchNum, channel*patchHeight*patchWidth + patchPixelY*patchWidth + patchPixelX, value);
                            }
                        }
                    }
                    patchNum++;
                }
            }
        }

        return output;
    }

    public static Matrix generatePatchesFromHiddenLayer(Matrix inputs, int inputWidth, int inputHeight, int patchWidth, int patchHeight) {
        // Input data has one row per example/pixel
        // Input data has one column per channel.

        // Input data is one row per example and one column per pixel.
        //      Assumes pixel-numbers are generated row-wise.
        //      (i.e. first all columns of row 0, then all columns of row 1 etc.)
        //      This is how the image-data is represented in the Kaggle Digits competition. (http://www.kaggle.com/c/digit-recognizer/data)
        // Output data have one row per example/patch
        // Output data have one column per channel/patchPixel

        int numPatchesPerExample = (inputWidth-patchWidth+1)*(inputHeight-patchHeight+1);
        int inputSize = inputHeight*inputWidth;
        int numExamples = inputs.numRows()/inputSize;
        int numChannels = inputs.numColumns();
        int patchSize = patchHeight*patchWidth;
        Matrix output = new Matrix(numExamples*numPatchesPerExample, numChannels*patchSize);
        for (int example = 0; example < numExamples; example++) {
            int patchNum = 0;
            for (int inputStartY = 0; inputStartY < inputHeight-patchHeight+1; inputStartY++) {
                for (int inputStartX = 0; inputStartX < inputWidth-patchWidth+1; inputStartX++) {
                    for (int channel = 0; channel < numChannels; channel++) {
                        for (int patchPixelY = 0; patchPixelY < patchHeight; patchPixelY++) {
                            for (int patchPixelX = 0; patchPixelX < patchWidth; patchPixelX++) {
                                int inputY = inputStartY + patchPixelY;
                                int inputX = inputStartX + patchPixelX;
                                double value = inputs.get(example*inputSize + inputY*inputWidth + inputX, channel);
                                output.set(example*numPatchesPerExample + patchNum, channel*patchSize + patchPixelY*patchWidth + patchPixelX, value);
                            }
                        }
                    }
                    patchNum++;
                }
            }
        }

        return output;
    }

    public static Matrix antiPatchDeltas(Matrix output, int inputWidth, int inputHeight, int patchWidth, int patchHeight) {
        // Input data has one row per example/pixel
        // Input data has one column per channel.

        // Input data is one row per example and one column per pixel.
        //      Assumes pixel-numbers are generated row-wise.
        //      (i.e. first all columns of row 0, then all columns of row 1 etc.)
        //      This is how the image-data is represented in the Kaggle Digits competition. (http://www.kaggle.com/c/digit-recognizer/data)
        // Output data have one row per example/patch
        // Output data have one column per channel/patchPixel

        int numPatchesPerExample = (inputWidth-patchWidth+1)*(inputHeight-patchHeight+1);
        int inputSize = inputHeight*inputWidth;
        int numExamples = output.numRows()/numPatchesPerExample;
        int patchSize = patchHeight*patchWidth;
        int numChannels = output.numColumns()/patchSize;
        Matrix inputs = new Matrix(numExamples*inputSize, numChannels);
        for (int example = 0; example < numExamples; example++) {
            int patchNum = 0;
            for (int inputStartY = 0; inputStartY < inputHeight-patchHeight+1; inputStartY++) {
                for (int inputStartX = 0; inputStartX < inputWidth-patchWidth+1; inputStartX++) {
                    for (int channel = 0; channel < numChannels; channel++) {
                        for (int patchPixelY = 0; patchPixelY < patchHeight; patchPixelY++) {
                            for (int patchPixelX = 0; patchPixelX < patchWidth; patchPixelX++) {
                                int inputY = inputStartY + patchPixelY;
                                int inputX = inputStartX + patchPixelX;
                                double value = output.get(example * numPatchesPerExample + patchNum, channel * patchSize + patchPixelY * patchWidth + patchPixelX);
                                inputs.set(example * inputSize + inputY * inputWidth + inputX, channel, inputs.get(example * inputSize + inputY * inputWidth + inputX, channel) + value);
                            }
                        }
                    }
                    patchNum++;
                }
            }
        }

        return inputs;
    }

    public static Matrix movePatchesToColumns(Matrix inputs, int numExamples, int numFeatureMaps, int numPatches) {
        Matrix output = new Matrix(numExamples, numFeatureMaps*numPatches);
        for (int example = 0; example < numExamples; example++) {
            for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
                for (int patch = 0; patch < numPatches; patch++) {
                    double value = inputs.get(example*numPatches + patch, featureMap);
                    output.set(example, featureMap*numPatches + patch, value);
                }
            }
        }
        return output;
    }

    public static Matrix movePatchesToRows(Matrix x, int numExamples, int numFeatureMaps, int numPatches) {
        Matrix output = new Matrix(numExamples*numPatches, numFeatureMaps);
        for (int example = 0; example < numExamples; example++) {
            for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
                for (int patch = 0; patch < numPatches; patch++) {
                    double value = x.get(example, featureMap*numPatches + patch);
                    output.set(example*numPatches + patch, featureMap, value);
                }
            }
        }
        return output;
    }

    public static class PoolingResult {
        public Matrix pooledActivations = null;
        public Matrix prePoolRowIndexes = null;

        public PoolingResult(Matrix pooledActivations, Matrix prePoolRowIndexes) {
            this.pooledActivations = pooledActivations;
            this.prePoolRowIndexes = prePoolRowIndexes;
        }
    }
}
