package weka.classifiers.functions;

import amten.ml.NNParams;
import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixElement;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

/**
 * Weka Classifier wrapper around NeuralNetwork class.
 *
 * (Convolutional) Neural network implementation with dropout and rectified linear units.
 * Can perform regression or classification.
 * Training is done by multithreaded mini-batch gradient descent with native matrix lib.
 *
 * @author Johannes Amtén
 *
 */

public class NeuralNetwork extends AbstractClassifier implements Serializable {

    // Parameters.
    private NNParams myParams = new NNParams();

    // Model
    private amten.ml.NeuralNetwork myNN = null;

    public NeuralNetwork() {
        super();
        // Want debug to be true by default, so user can see training cost for each iteration and halt training.
        setDebug(true);
    }

    public void buildClassifier(Instances instances) throws Exception {

        int numExamples = instances.numInstances();
        int numInputAttributes = instances.numAttributes() - 1;

        int classIndex = instances.classIndex();
        int numClasses = instances.numClasses();

        // Get class values, y.
        double[] classValues = instances.attributeToDoubleArray(classIndex);
        Matrix y = new Matrix(numExamples, 1);
        for (MatrixElement me: y) {
            me.set(classValues[me.row()]);
        }


        // Get input values, x.

        // Find nominal columns and their number of categories
        Matrix x = new Matrix(numExamples, numInputAttributes);
        int[] numCategories = new int[numInputAttributes];
        int col = 0;
        for (int attrIndex = 0; attrIndex < instances.numAttributes(); attrIndex++) {
            Attribute attr = instances.attribute(attrIndex);
            if (attrIndex != classIndex) {
                for (int row = 0; row < numExamples; row++) {
                    double value = instances.get(row).value(attrIndex);
                    boolean missing = instances.get(row).isMissing(attrIndex);
                    if (missing) {
                        value = attr.isNominal() ? -1.0 : 0.0;
                    }
                    x.set(row, col, value);
                }
                // Find number of categories of nominal column.
                numCategories[col] = attr.isNominal() ?  attr.numValues() : 1;
                col++;
            }
        }

        myParams.numClasses = numClasses;
        myParams.numCategories = numCategories;

        myNN = new amten.ml.NeuralNetwork(myParams);
        myNN.train(x, y);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {

        Matrix x = new Matrix(1, instance.numAttributes()-1);
        int classIndex = instance.classIndex();

        int col = 0;
        for (int attrIndex = 0; attrIndex < instance.numAttributes(); attrIndex++) {
            Attribute attr = instance.attribute(attrIndex);
            if (attrIndex != classIndex) {
                double value = instance.value(attrIndex);
                boolean missing = instance.isMissing(attrIndex);
                if (missing) {
                    value = attr.isNominal() ? -1.0 : 0.0;
                }
                x.set(0, col, value);
                col++;
            }
        }

        return myNN.getPredictions(x).getRow(0);
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        ArrayList<Option> options = new ArrayList<>();

        options.add(new Option(
                "\tNumber of examples in each mini-batch.",
                "BatchSize", 1, "-bs"));
        options.add(new Option(
                "\tWeight penalty",
                "WeightPenalty", 1, "-wp"));
        options.add(new Option(
                "\tLearning rate",
                "LearningRate", 1, "-lr"));
        options.add(new Option(
                "\tMaximum number of training iterations over the entire data set. (epochs)",
                "MaxIterations", 1, "-mi"));
        options.add(new Option(
                "\tNumber of threads to use for training the network.",
                "Threads", 1, "-th"));
        options.add(new Option(
                "\tNumber of Units in the hidden layers. (comma-separated list)\n" +
                "e.g. \"100,100\" for two layers with 100 units each.\n" +
                "For convolutional layers: <num feature maps>-<patch-width>-<patch-height>-<pool-width>-<pool-height> \n" +
                "e.g. \"20-5-5-2-2,100-5-5-2-2\" for two convolutional layers, both with patch size 5x5 and pool size 2x2, each with 20 and 100 feature maps respectively.",
                "HiddenLayers", 1, "-hl"));
        options.add(new Option(
                "\tFraction of units to dropout in the input layer during training.",
                "InputLayerDropoutRate", 1, "-di"));
        options.add(new Option(
                "\tFraction of units to dropout in the hidden layers during training.",
                "HiddenLayersDropoutRate", 1, "-dh"));
        options.add(new Option(
                "\tWidth of input image (only used for convolution) (0=Square image).",
                "InputWidth", 1, "-iw"));

        return Collections.enumeration(options);
    }

    public void setOptions(String[] options) throws Exception {

        String weightPenaltyString = Utils.getOption("wp", options);
        myParams.weightPenalty = weightPenaltyString.equals("") ? myParams.weightPenalty : Double.parseDouble(weightPenaltyString);

        String lrString = Utils.getOption("lr", options);
        myParams.learningRate = lrString.equals("") ? myParams.learningRate : Double.parseDouble(lrString);

        String maxIterationsString = Utils.getOption("mi", options);
        myParams.maxIterations = maxIterationsString.equals("") ? myParams.maxIterations : Integer.parseInt(maxIterationsString);

        String threadsString = Utils.getOption("th", options);
        myParams.numThreads = threadsString.equals("") ? myParams.numThreads : Integer.parseInt(threadsString);

        String batchSizeString = Utils.getOption("bs", options);
        myParams.batchSize = batchSizeString.equals("") ? myParams.batchSize : Integer.parseInt(batchSizeString);

        String hiddenLayersString = Utils.getOption("hl", options);
        myParams.hiddenLayerParams = hiddenLayersString.equals("") ? myParams.hiddenLayerParams  : getHiddenLayers(hiddenLayersString);

        String inputLayerDropoutRateString = Utils.getOption("di", options);
        myParams.inputLayerDropoutRate = inputLayerDropoutRateString.equals("") ? myParams.inputLayerDropoutRate : Double.parseDouble(inputLayerDropoutRateString);

        String hiddenLayersDropoutRateString = Utils.getOption("dh", options);
        myParams.hiddenLayersDropoutRate = hiddenLayersDropoutRateString.equals("") ? myParams.hiddenLayersDropoutRate : Double.parseDouble(hiddenLayersDropoutRateString);

        String inputWidthString = Utils.getOption("iw", options);
        myParams.inputWidth = inputWidthString.equals("") ? myParams.inputWidth : Integer.parseInt(inputWidthString);
    }

    public String [] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        options.add("-lr");
        options.add(Double.toString(myParams.learningRate));
        options.add("-wp");
        options.add(Double.toString(myParams.weightPenalty));
        options.add("-mi");
        options.add(Integer.toString(myParams.maxIterations));
        options.add("-bs");
        options.add(Integer.toString(myParams.batchSize));
        options.add("-th");
        options.add(Integer.toString(myParams.numThreads));
        options.add("-hl");
        options.add(getString(myParams.hiddenLayerParams));
        options.add("-di");
        options.add(Double.toString(myParams.inputLayerDropoutRate));
        options.add("-dh");
        options.add(Double.toString(myParams.hiddenLayersDropoutRate));
        options.add("-iw");
        options.add(Integer.toString(myParams.inputWidth));

        return options.toArray(new String[options.size()]);
    }

    public double getWeightPenalty() {
        return myParams.weightPenalty;
    }
    public void setWeightPenalty(double weightPenalty) {
        myParams.weightPenalty = weightPenalty;
    }
    public String weightPenaltyTipText() {
        return "Weight penalty parameter.";
    }

    public String getHiddenLayers() {
        return getString(myParams.hiddenLayerParams);
    }
    public void setHiddenLayers(String hiddenLayers) {
        myParams.hiddenLayerParams = getHiddenLayers(hiddenLayers);
    }
    public String hiddenLayersTipText() {
        return "Number of units in each hidden layer (comma-separated) (For convolutional layers: <num feature maps>-<patch-width>-<patch-height>-<pool-width>-<pool-height>).";
    }

    public int getMaxIterations() {
        return myParams.maxIterations;
    }
    public void setMaxIterations(int iterations) {
        myParams.maxIterations = iterations;
    }
    public String maxIterationsTipText() {
        return "Maximum number of training iterations over the entire data set (epochs)";
    }

    public double getInputLayerDropoutRate() {
        return myParams.inputLayerDropoutRate;
    }
    public void setInputLayerDropoutRate(double inputLayerDropoutRate) {
        myParams.inputLayerDropoutRate = inputLayerDropoutRate;
    }
    public String inputLayerDropoutRateTipText() {
        return "Fraction of units to dropout in the input layer during training.";
    }

    public double getHiddenLayersDropoutRate() {
        return myParams.hiddenLayersDropoutRate;
    }
    public void setHiddenLayersDropoutRate(double hiddenLayersDropoutRate) {
        myParams.hiddenLayersDropoutRate = hiddenLayersDropoutRate;
    }
    public String hiddenLayersDropoutRateTipText() {
        return "Fraction of units to dropout in the hidden layers during training.";
    }

    public int getBatchSize() {
        return myParams.batchSize;
    }
    public void setBatchSize(int batchSize) {
        myParams.batchSize = batchSize;
    }
    public String batchSizeTipText() {
        return "Number of training examples in each mini-batch (0=Auto-choose) .";
    }

    public int getThreads() {
        return myParams.numThreads;
    }
    public void setThreads(int threads) {
        myParams.numThreads = threads;
    }
    public String threadsTipText() {
        return "The number of threads to use for training the network (0=Auto-detect)";
    }

    public double getLearningRate() {
        return myParams.learningRate;
    }
    public void setLearningRate(double learningRate) {
        myParams.learningRate = learningRate;
    }
    public String learningRateTipText() {
        return "Learning rate (0=Auto-detect).";
    }

    public int getInputWidth() {
        return myParams.inputWidth;
    }
    public void setInputWidth(int width) {
        myParams.inputWidth = width;
    }
    public String inputWidthTipText() {
        return "Width of input image (only used for convolution) (0=Square image).";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        // result.enable(Capabilities.Capability.DATE_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        // result.enable(Capabilities.Capability.DATE_CLASS);
        // result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    private NNParams.NNLayerParams[] getHiddenLayers(String s) {
        String[] stringList = s.split(",");
        ArrayList<NNParams.NNLayerParams> layerList = new ArrayList<>();
        for (String layerString:stringList) {
            if (layerString.contains("-")) {
                // Convolutional layer
                String[] convStringList = layerString.split("-");
                if (convStringList.length >=3) {
                    int numFeatureMaps = Integer.parseInt(convStringList[0]);
                    int patchWidth = Integer.parseInt(convStringList[1]);
                    int patchHeight = Integer.parseInt(convStringList[2]);
                    int poolWidth = convStringList.length > 4 ? Integer.parseInt(convStringList[3]) : 0;
                    int poolHeight = convStringList.length > 4 ? Integer.parseInt(convStringList[4]) : 0;
                    layerList.add(new NNParams.NNLayerParams(numFeatureMaps, patchWidth, patchHeight, poolWidth, poolHeight));
                }
            } else if (!layerString.equals("")) {
                layerList.add(new NNParams.NNLayerParams(Integer.parseInt(layerString)));
            }
        }
        return layerList.toArray(new NNParams.NNLayerParams[layerList.size()]);
    }

    private String getString(NNParams.NNLayerParams[] layerList) {
        String s = "";
        for (NNParams.NNLayerParams layer: layerList) {
            if (!s.equals("")) {
                s += ",";
            }
            if (layer.isConvolutional()) {
                s += layer.numFeatures + "-" + layer.patchWidth + "-" + layer.patchHeight;
                if (layer.isPooled()) {
                    s += "-" + layer.poolWidth + "-" + layer.poolHeight;
                }
            } else {
                s += layer.numFeatures;
            }
        }
        return s;
    }

    public String globalInfo() {
        return "(Convolutional) Neural Network implementation with dropout regularization and Rectified Linear Units.\n\n" +
                "Training is done with multithreaded mini-batch gradient descent.\n\n" +
                "Running Weka with console window and with debug flag for this classifier on, you can monitor training cost in console window and halt training anytime by pressing enter.\n\n" +
                "Hidden layers are specified as comma-separated lists.\n" +
                "e.g. \"100,100\" for two layers with 100 units each.\n" +
                "For convolutional layers: <num feature maps>-<patch-width>-<patch-height>-<pool-width>-<pool-height> \n" +
                "e.g. \"20-5-5-2-2,100-5-5-2-2\" for two convolutional layers, both with patch size 5x5 and pool size 2x2, each with 20 and 100 feature maps respectively.";
    }

    public static void main(String [] argv){
        runClassifier(new NeuralNetwork(), argv);
    }
}