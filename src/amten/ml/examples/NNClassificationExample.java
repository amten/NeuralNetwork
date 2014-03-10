package amten.ml.examples;

import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixUtils;

/**
 * Examples of using NeuralNetwork for classification.
 *
 * @author Johannes Amtén
 */
public class NNClassificationExample {

    /**
     * Performs classification of Handwritten digits,
     * using a subset (1000 rows) from the Kaggle Digits competition.
     * <br></br>
     * Uses file /example_data/Kaggle_Digits_1000.csv
     *
     * @see <a href="http://www.kaggle.com/c/digit-recognizer">http://www.kaggle.com/c/digit-recognizer</a></a>
     */
    public static void runKaggleDigitsClassification() throws Exception {
        System.out.println("Running classification on Kaggle Digits dataset...\n");
        // Read data from CSV-file
        int headerRows = 1;
        char separator = ',';
        Matrix data = MatrixUtils.readCSV("example_data/Kaggle_Digits_1000.csv", separator, headerRows);

        // Split data into training set and crossvalidation set.
        float crossValidationPercent = 33;
        Matrix[] split = MatrixUtils.split(data, crossValidationPercent, 0);
        Matrix dataTrain = split[0];
        Matrix dataCV = split[1];

        // First column contains the classification label. The rest are the indata.
        Matrix xTrain = dataTrain.getColumns(1, -1);
        Matrix yTrain = dataTrain.getColumns(0, 0);
        Matrix xCV = dataCV.getColumns(1, -1);
        Matrix yCV = dataCV.getColumns(0, 0);

        int[] numCategories = null; // Just numeric indata, no nominal attributes.
        int numClasses = 10; // 10 digits to classify
        int[] hiddenUnits = { 100 };
        double weightPenalty = 1E-8;
        // Learning rate 0 will autofind an initial learning rate.
        double learningRate = 0;
        int batchSize = 100;
        int iterations = 200;
        // Threads = 0 will autofind number of processor cores in computer.
        int threads = 0;
        double inputLayerDropoutRate = 0.2;
        double hiddenLayersDropoutRate = 0.5;
        boolean debug = true;
        boolean normalize = true;

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork();
        nn.train(xTrain, numCategories, yTrain, numClasses, hiddenUnits, weightPenalty, learningRate, batchSize, iterations, threads, inputLayerDropoutRate, hiddenLayersDropoutRate, debug, normalize);
        System.out.println("\nTraining time: " + String.format("%.3g", (System.currentTimeMillis() - startTime) / 1000.0) + "s");

        int[] predictedClasses = nn.getPredictedClasses(xTrain);
        int correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yTrain.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Training set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");

        predictedClasses = nn.getPredictedClasses(xCV);
        correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yCV.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Crossvalidation set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");
    }

    /**
     * Performs classification of titanic survivors/casualties,
     * using a cleaned dataset from the Kaggle Digits competition.
     * <br></br>
     * Dataset have been cleaned by removing some string attributes,
     * converting some string attributes to nominal (replacing string values with numeric indexes)
     * and by filling in missing values with mean/mode values.
     * <br></br>
     * Uses file /example_data/Kaggle_Titanic_cleaned.csv
     *
     * @see <a href="http://www.kaggle.com/c/titanic-gettingStarted">http://www.kaggle.com/c/titanic-gettingStarted</a></a>
     */
    public static void runKaggleTitanicClassification() throws Exception {
        System.out.println("Running classification on Kaggle Titanic dataset...\n");
        // Read data from CSV-file
        int headerRows = 1;
        char separator = ',';
        Matrix data = MatrixUtils.readCSV("example_data/Kaggle_Titanic_Cleaned.csv", separator, headerRows);

        // Split data into training set and crossvalidation set.
        float crossValidationPercent = 33;
        Matrix[] split = MatrixUtils.split(data, crossValidationPercent, 0);
        Matrix dataTrain = split[0];
        Matrix dataCV = split[1];

        // First column contains the classification label. The rest are the indata.
        Matrix xTrain = dataTrain.getColumns(1, -1);
        Matrix yTrain = dataTrain.getColumns(0, 0);
        Matrix xCV = dataCV.getColumns(1, -1);
        Matrix yCV = dataCV.getColumns(0, 0);

        // Pclass has 3 categories
        // Sex has 2 categories
        // Embarked has 3 categories
        // The rest of the attributes are numeric (as indicated with "1").
        int[] numCategories = {3, 2, 1, 1, 1, 1, 3};

        int numClasses = 2; // 2 classes, survived/not
        int[] hiddenUnits = { 100 };
        double weightPenalty = 1E-8;
        // Learning rate 0 will autofind an initial learning rate.
        double learningRate = 0;
        int batchSize = 100;
        int iterations = 200;
        // Threads = 0 will autofind number of processor cores in computer.
        int threads = 0;
        double inputLayerDropoutRate = 0.2;
        double hiddenLayersDropoutRate = 0.5;
        boolean debug = true;
        boolean normalize = true;

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork();
        nn.train(xTrain, numCategories, yTrain, numClasses, hiddenUnits, weightPenalty, learningRate, batchSize, iterations, threads, inputLayerDropoutRate, hiddenLayersDropoutRate, debug, normalize);
        System.out.println("\nTraining time: " + String.format("%.3g", (System.currentTimeMillis() - startTime) / 1000.0) + "s");

        int[] predictedClasses = nn.getPredictedClasses(xTrain);
        int correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yTrain.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Training set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");

        predictedClasses = nn.getPredictedClasses(xCV);
        correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yCV.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Crossvalidation set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");
    }

    public static void main(String[] args) throws Exception {
        runKaggleDigitsClassification();
        System.out.println("\n\n\n");
        runKaggleTitanicClassification();
    }
}
