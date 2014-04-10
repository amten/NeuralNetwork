package amten.ml.test;

import amten.ml.NNParams;
import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Johannes Amtén on 2014-02-24.
 *
 */
public class NeuralNetworkTest {


    /**
     * Performs classification of Handwritten digits,
     * using a subset (1000 rows) from the Kaggle Digits competition.
     */
    public static void runKaggleDigitsClassification() throws Exception {
        // Read data from CSV-file
        int headerRows = 1;
        char separator = ',';
        Matrix data = MatrixUtils.readCSV("example_data/Kaggle_Digits_1000.csv", separator, headerRows);
//        Matrix data = MatrixUtils.readCSV("C:/Users/Glenn/Documents/Kaggle/Digit Recognizer/train.csv", separator, headerRows);

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
        params.hiddenLayerParams = new NNParams.NNLayerParams[] { new NNParams.NNLayerParams(20, 5, 5, 2, 2) , new NNParams.NNLayerParams(100, 5, 5, 2, 2) };
        params.learningRate = 1E-2;
        params.maxIterations = 10;

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork(params);
        nn.train(xTrain, yTrain);
        System.out.println("Training time: " + (System.currentTimeMillis() - startTime) / 1000.0 + "s");

        List<Matrix> batchesX = new ArrayList<>();
        List<Matrix> batchesY = new ArrayList<>();
        MatrixUtils.split(xTrain, yTrain, params.batchSize, batchesX, batchesY);
        int correct = 0;
        for (int batch = 0; batch < batchesX.size(); batch++) {
            int[] predictedClasses = nn.getPredictedClasses(batchesX.get(batch));
            for (int i = 0; i < predictedClasses.length; i++) {
                if (predictedClasses[i] == batchesY.get(batch).get(i, 0)) {
                    correct++;
                }
            }
        }
        System.out.println("Training set accuracy: " + (double) correct/xTrain.numRows()*100 + "%");

        batchesX = new ArrayList<>();
        batchesY = new ArrayList<>();
        MatrixUtils.split(xCV, yCV, params.batchSize, batchesX, batchesY);
        correct = 0;
        for (int batch = 0; batch < batchesX.size(); batch++) {
            int[] predictedClasses = nn.getPredictedClasses(batchesX.get(batch));
            for (int i = 0; i < predictedClasses.length; i++) {
                if (predictedClasses[i] == batchesY.get(batch).get(i, 0)) {
                    correct++;
                }
            }
        }
        System.out.println("Crossvalidation set accuracy: " + (double) correct/xCV.numRows()*100 + "%");

    }

/////////////// Gradient test
//
//    public static void main(String[] args) throws Exception {
//        // Read data from CSV-file
//        int headerRows = 1;
//        char separator = ',';
//        Matrix data = MatrixUtils.readCSV("example_data/Kaggle_Digits_1000.csv", separator, headerRows);
//
//
//        // First column contains the classification label. The rest are the indata.
//        Matrix x = data.getColumns(1, -1);
//        Matrix y = data.getColumns(0, 0);
//
//        int numClasses = 10; // 10 digits to classify
//        int intputWidth = 0;
//        NNLayerParams[] hiddenLayerParams = { new NNLayerParams(20, 5, 5, 2, 2) , new NNLayerParams(100, 5, 5, 2, 2) };
//        double weightPenalty = 1E-8;
//        // Learning rate 0 will autofind an initial learning rate.
//        double learningRate = 1e-3; //0.0;
//        int batchSize = 1;
//        int iterations = 1;
//        // Threads = 0 will autofind number of processor cores in computer.
//        int threads = 1;
//        double inputLayerDropoutRate = 0.0;
//        double hiddenLayersDropoutRate = 0.0;
//        boolean debug = true;
//        boolean normalize = true;
//
//        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork();
//        nn.train(x, null, y, numClasses, intputWidth, hiddenLayerParams, weightPenalty, learningRate, batchSize, iterations, threads, inputLayerDropoutRate, hiddenLayersDropoutRate, debug, normalize);
//
//        y = MatrixUtils.expandNominalAttributes(y, new int[] {nn.myNumClasses} );
//        double cost = nn.getCost(x, y, weightPenalty, x.numRows());
//        Matrix[] grads = nn.getGradients(x, y, weightPenalty, null, x.numRows());
//        for (int t = 1; t < nn.myThetas.length; t++) {
//            Matrix theta = nn.myThetas[t];
//            Matrix grad = grads[t];
//            System.out.println("theta " + t);
//            for (int row = 0; row < theta.numRows(); row++) {
//                for (int col = 0; col < theta.numColumns(); col++) {
//                    double value = theta.get(row, col);
//                    double delta = value*1.0E-5;
//                    theta.set(row, col, value + delta);
//                    double cost2 = nn.getCost(x, y, weightPenalty, x.numRows());
//                    double gradDiff = (cost2-cost)/delta - grad.get(row, col);
//                    if (Math.abs(gradDiff) > Math.abs(grad.get(row, col)/100.0)) {
//                        System.out.println("    " + gradDiff + "    " + (cost2-cost)/delta + "    " + grad.get(row, col) + "    (" + value + ")");
//                    } else {
//                        //                        System.out.println(gradDiff + "    " + (cost2-cost)/delta + "    " + grad.get(row, col) + "    (" + value + ")");
//                    }
//                    theta.set(row, col, value);
//                }
//
//            }
//        }
//    }

    public static void main(String[] args) throws Exception {
        runKaggleDigitsClassification();

    }
}
