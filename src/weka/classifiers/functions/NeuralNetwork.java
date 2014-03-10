package weka.classifiers.functions;

import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixElement;
import amten.ml.matrix.MatrixUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

/**
 * Weka Classifier wrapper around NeuralNetwork class.
 *
 * Neural network implementation with dropout and rectified linear units.
 * Can perform regression or classification.
 * Training is done by multithreaded mini-batch gradient descent with native matrix lib.

 *
 */

public class NeuralNetwork extends AbstractClassifier implements Serializable {


    // Classifier parameters
    private double myWeightPenalty = 1E-8;
    private double myLearningRate = 0.0;
    private int[] myHiddenUnits = { 100 };
    private int myBatchSize = 100;
    private int myIterations = 200;
    private int myThreads = 0;
    private double myInputLayerDropoutRate = 0.2;
    private double myHiddenLayersDropoutRate = 0.5;

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

        myNN = new amten.ml.NeuralNetwork();
        myNN.train(x, numCategories, y, numClasses, myHiddenUnits, myWeightPenalty, myLearningRate, myBatchSize, myIterations, myThreads, myInputLayerDropoutRate, myHiddenLayersDropoutRate, getDebug(), true);
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
                "\tNumber of training iterations over the entire data set. (epochs)",
                "Iterations", 1, "-i"));
        options.add(new Option(
                "\tNumber of threads to use for training the network.",
                "Threads", 1, "-t"));
        options.add(new Option(
                "\tNumber of Units in the hidden layers. (comma-separated list)",
                "HiddenUnits", 1, "-hu"));
        options.add(new Option(
                "\tFraction of units to dropout in the input layer during training.",
                "InputLayerDropoutRate", 1, "-di"));
        options.add(new Option(
                "\tFraction of units to dropout in the hidden layers during training.",
                "HiddenLayersDropoutRate", 1, "-dh"));

        return Collections.enumeration(options);
    }

    public void setOptions(String[] options) throws Exception {

        String weightPenaltyString = Utils.getOption("wp", options);
        myWeightPenalty = weightPenaltyString.equals("") ? 0.0 : Double.parseDouble(weightPenaltyString);
        String lrString = Utils.getOption("lr", options);
        myLearningRate = lrString.equals("") ? 0.0 : Double.parseDouble(lrString);
        String iterationsString = Utils.getOption('i', options);
        myIterations = iterationsString.equals("") ? 10000 : Integer.parseInt(iterationsString);
        String threadsString = Utils.getOption('t', options);
        myThreads = threadsString.equals("t") ? 0 : Integer.parseInt(threadsString);
        String batchSizeString = Utils.getOption("bs", options);
        myBatchSize = batchSizeString.equals("") ? 100 : Integer.parseInt(batchSizeString);
        String hiddenUnitsString = Utils.getOption("hu", options);
        myHiddenUnits = hiddenUnitsString.equals("") ? new int[] { 50, 50}  : getIntList(hiddenUnitsString);
        String inputLayerDropoutRateString = Utils.getOption("di", options);
        myInputLayerDropoutRate = inputLayerDropoutRateString.equals("") ? 0.2 : Double.parseDouble(inputLayerDropoutRateString);
        String hiddenLayersDropoutRateString = Utils.getOption("dh", options);
        myHiddenLayersDropoutRate = hiddenLayersDropoutRateString.equals("") ? 0.5 : Double.parseDouble(hiddenLayersDropoutRateString);
    }

    public String [] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        options.add("-lr");
        options.add(Double.toString(myLearningRate));
        options.add("-wp");
        options.add(Double.toString(myWeightPenalty));
        options.add("-i");
        options.add(Integer.toString(myIterations));
        options.add("-bs");
        options.add(Integer.toString(myBatchSize));
        options.add("-t");
        options.add(Integer.toString(myThreads));
        options.add("-hu");
        options.add(getString(myHiddenUnits));
        options.add("-di");
        options.add(Double.toString(myInputLayerDropoutRate));
        options.add("-dh");
        options.add(Double.toString(myHiddenLayersDropoutRate));
        return options.toArray(new String[options.size()]);
    }

    public double getWeightPenalty() {
        return myWeightPenalty;
    }
    public void setWeightPenalty(double weightPenalty) {
        myWeightPenalty = weightPenalty;
    }
    public String weightPenaltyTipText() {
        return "Weight penalty parameter.";
    }

    public String getHiddenUnits() {
        return getString(myHiddenUnits);
    }
    public void setHiddenUnits(String hiddenUnits) {
        myHiddenUnits = getIntList(hiddenUnits);
    }
    public String hiddenUnitsTipText() {
        return "Number of units in each hidden layer (comma-separated).";
    }

    public int getIterations() {
        return myIterations;
    }
    public void setIterations(int iterations) {
        myIterations = iterations;
    }
    public String iterationsTipText() {
        return "Number of training iterations over the entire data set (epochs)";
    }

    public double getInputLayerDropoutRate() {
        return myInputLayerDropoutRate;
    }
    public void setInputLayerDropoutRate(double inputLayerDropoutRate) {
        myInputLayerDropoutRate = inputLayerDropoutRate;
    }
    public String inputLayerDropoutRateTipText() {
        return "Fraction of units to dropout in the input layer during training.";
    }

    public double getHiddenLayersDropoutRate() {
        return myHiddenLayersDropoutRate;
    }
    public void setHiddenLayersDropoutRate(double hiddenLayersDropoutRate) {
        myHiddenLayersDropoutRate = hiddenLayersDropoutRate;
    }
    public String hiddenLayersDropoutRateTipText() {
        return "Fraction of units to dropout in the hidden layers during training.";
    }

    public int getBatchSize() {
        return myBatchSize;
    }
    public void setBatchSize(int batchSize) {
        myBatchSize = batchSize;
    }
    public String batchSizeTipText() {
        return "Number of training examples in each mini-batch.";
    }

    public int getThreads() {
        return myThreads;
    }
    public void setThreads(int threads) {
        myThreads = threads;
    }
    public String threadsTipText() {
        return "The number of threads to use for training the network (0=Auto-detect)";
    }

    public double getLearningRate() {
        return myLearningRate;
    }
    public void setLearningRate(double learningRate) {
        myLearningRate = learningRate;
    }
    public String learningRateTipText() {
        return "Learning rate (0=Auto-detect).";
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

    private int[] getIntList(String s) {
        String[] stringList = s.split(",");
        int[] intList = new int[stringList.length];
        for (int i = 0; i < stringList.length ; i++) {
            String intString = stringList[i];
            intList[i] = intString.equals("") ? 0 : Integer.parseInt(intString);
        }
        return intList;
    }

    private String getString(int[] intList) {
        String s = "";
        for (int i: intList) {
            if (!s.equals("")) {
                s += ",";
            }
            s += i;
        }
        return s;
    }

    public String globalInfo() {
        return "Neural Network implementation with dropout regularization and Rectified Linear Units.\n" +
                "Training is done with multithreaded mini-batch gradient descent.\n" +
                "Running Weka with console window and with debug flag for this classifier on, you can monitor training cost in console window and halt training anytime by pressing enter.";
    }

}