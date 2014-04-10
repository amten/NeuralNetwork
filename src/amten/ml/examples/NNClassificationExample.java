package amten.ml.examples;

import amten.ml.NNParams;
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
    public static void runKaggleDigitsClassification(boolean useConvolution) throws Exception {
        if (useConvolution) {
            System.out.println("Running classification on Kaggle Digits dataset, with convolution...\n");
        } else {
            System.out.println("Running classification on Kaggle Digits dataset...\n");
        }
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

        NNParams params = new NNParams();
        params.numClasses = 10; // 10 digits to classify
        params.hiddenLayerParams = useConvolution ? new NNParams.NNLayerParams[]{ new NNParams.NNLayerParams(20, 5, 5, 2, 2) , new NNParams.NNLayerParams(100, 5, 5, 2, 2) } :
                                                    new NNParams.NNLayerParams[] { new NNParams.NNLayerParams(100) };
        params.maxIterations = useConvolution ? 10 : 200;
        params.learningRate = useConvolution ? 1E-2 : 0;

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork(params);
        nn.train(xTrain, yTrain);
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

        NNParams params = new NNParams();
        // Pclass has 3 categories
        // Sex has 2 categories
        // Embarked has 3 categories
        // The rest of the attributes are numeric (as indicated with "1").
        params.numCategories = new int[]  {3, 2, 1, 1, 1, 1, 3};
        params.numClasses = 2; // 2 classes, survived/not

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork(params);
        nn.train(xTrain, yTrain);
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
        runKaggleDigitsClassification(false);
        System.out.println("\n\n\n");
        runKaggleDigitsClassification(true);
        System.out.println("\n\n\n");
        runKaggleTitanicClassification();
    }
}
