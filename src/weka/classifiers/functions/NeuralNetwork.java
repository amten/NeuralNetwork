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
 * Created by Johannes Amtén on 2014-02-24.
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
    private double[] myAverages = null;
    private double[] myStdDevs = null;
    ArrayList<Integer> myNumericColumns;
    ArrayList<Integer> myNominalColumns;
    int myNumNominalCategories = 0;

    public NeuralNetwork() {
        super();
        // Want debug to be true by default, so user can see training cost for each iteration and halt training.
        setDebug(true);
    }

    public void buildClassifier(Instances instances) throws Exception {

        int numExamples = instances.numInstances();

        int classIndex = instances.classIndex();
        int numClasses = instances.numClasses();

        // Get class values, y.
        boolean softmax = instances.attribute(classIndex).isNominal();
        double[] classValues = instances.attributeToDoubleArray(classIndex);
        Matrix y;
        if (softmax) {
            // Change answers from single nominal to group of booleans
            y = new Matrix(numExamples, numClasses);
            for (MatrixElement me: y) {
                me.set(me.col() ==  classValues[me.row()] ? 1.0 : 0.0 );
            }
        } else {
            y = new Matrix(numExamples, 1);
            for (MatrixElement me: y) {
                me.set(classValues[me.row()]);
            }
        }


        // Get input values, x.

        // First, find numerical and nominal columns
        myNumericColumns = new ArrayList<>();
        myNominalColumns = new ArrayList<>();
        myNumNominalCategories = 0;
        for (int attrIndex = 0; attrIndex < instances.numAttributes(); attrIndex++) {
            Attribute attr = instances.attribute(attrIndex);
            if (attr.isNumeric() && attrIndex != classIndex) {
                myNumericColumns.add(attrIndex);
            } else if (attr.isNominal() && attrIndex != classIndex) {
                myNominalColumns.add(attrIndex);
                myNumNominalCategories += attr.numValues();
            }
        }

        // Get all numeric values and normalize them
        Matrix xNumeric = new Matrix(numExamples, myNumericColumns.size());
        int xCol = 0;
        for (int col:myNumericColumns) {
            for (int inst = 0; inst < numExamples; inst++) {
                boolean missing = instances.get(inst).isMissing(col);
                double value = missing ? 0.0 : instances.get(inst).value(col);
                xNumeric.set(inst, xCol, value);
            }
            xCol++;
        }
        myAverages = MatrixUtils.getAverages(xNumeric);
        myStdDevs = MatrixUtils.getStandardDeviations(xNumeric);
        MatrixUtils.normalizeData(xNumeric, myAverages, myStdDevs);

        // Get all nominal values and expand them to groups of booleans.
        Matrix xNominal = new Matrix(numExamples, myNumNominalCategories);
        xCol = 0;
        for (int col:myNominalColumns) {
            Attribute attr = instances.attribute(col);
            int numCategories = attr.numValues();
            // One unit for each category.
            for (int cat = 0; cat < numCategories; cat++) {
                for (int inst = 0; inst < numExamples; inst++) {
                    boolean missing = instances.get(inst).isMissing(col);
                    double value = !missing && cat == instances.get(inst).value(col) ? 1.0 : 0.0;
                    xNominal.set(inst, xCol, value);
                }
                xCol++;
            }
        }

        Matrix x = xNumeric.addColumns(xNominal);

        myNN = new amten.ml.NeuralNetwork();
        myNN.train(x, y, myHiddenUnits, myWeightPenalty, myLearningRate, myBatchSize, myIterations, myThreads, myInputLayerDropoutRate, myHiddenLayersDropoutRate, softmax, getDebug());
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        // First, get all numeric values and normalize them
        double[] xNumeric = new double[myNumericColumns.size()];
        int xIndex = 0;
        for (int col:myNumericColumns) {
            double value = instance.isMissing(col) ? 0.0 : instance.value(col);
            xNumeric[xIndex] = value;
            xIndex++;
        }
        MatrixUtils.normalizeData(xNumeric, myAverages, myStdDevs);

        // Then get all nominal values and expand them to groups of booleans.
        double[] x = new double[xNumeric.length + myNumNominalCategories];
        System.arraycopy(xNumeric, 0, x, 0, xNumeric.length);
        for (int col:myNominalColumns) {
            Attribute attr = instance.attribute(col);
            int numCategories = attr.numValues();
            // One unit for each category.
            for (int cat = 0; cat < numCategories; cat++) {
                boolean missing = instance.isMissing(col);
                double value = !missing && cat == instance.value(col) ? 1.0 : 0.0;
                x[xIndex] = value;
                xIndex++;
            }
        }

        return myNN.getPredictions(x);
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

        String lambdaString = Utils.getOption("wp", options);
        myWeightPenalty = lambdaString.equals("") ? 0.0 : Double.parseDouble(lambdaString);
        String alphaString = Utils.getOption("lr", options);
        myLearningRate = alphaString.equals("") ? 0.0 : Double.parseDouble(alphaString);
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