package amten.ml.test;

import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixElement;
import amten.ml.matrix.MatrixUtils;

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
        nn.train(xTrain, null, yTrain, numClasses, hiddenUnits, weightPenalty, learningRate, batchSize, iterations, threads, inputLayerDropoutRate, hiddenLayersDropoutRate, debug, normalize);
        System.out.println("Training time: " + (System.currentTimeMillis() - startTime) / 1000.0 + "s");

        int[] predictedClasses = nn.getPredictedClasses(xTrain);
        int correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yTrain.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Training set accuracy: " + (double) correct/predictedClasses.length*100 + "%");

        predictedClasses = nn.getPredictedClasses(xCV);
        correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yCV.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Crossvalidation set accuracy: " + (double) correct/predictedClasses.length*100 + "%");

    }

//    public static void runGradientTest() {
//        DenseMatrix x = MatrixUtils.readCSV("C:\\Users\\glenn\\Documents\\Octave\\ML\\ex4\\x.csv", ',', 0);
//        DenseVector y = MatrixUtils.getSubVector(MatrixUtils.readCSV("C:\\Users\\glenn\\Documents\\Octave\\ML\\ex4\\y.csv", ',', 0),  0, -1, 0);
//
//        // Change answers from nominal to 10 booleans
//        DenseMatrix yExpanded = new DenseMatrix(x.numRows(), 10);
//        for (MatrixEntry me: yExpanded)
//        {
//            me.set(me.column() + 1 ==  y.get(me.row()) ? 1 : 0  );
//        }
//
//        int[] hiddenUnits = { 50, 50 };
//        double lambda = 0.0;
//        double alpha = 1E-6;
//        int batchSize = 100;
//        int iterations = 1;
//        int threads = 0;
//        boolean dropout = false;
//        boolean softmax = true;
//        double momentum = 0.0;
//        DenseMatrix[] thetas = computeThetas(x, yExpanded, hiddenUnits, lambda, alpha, batchSize, iterations, threads, dropout, softmax, momentum);
//
////        softmax = false;
//		DenseMatrix xWithBias = MatrixUtils.getSubMatrix(MatrixUtils.addBiasColumn(x), 0, 0, 0, -1);
//        yExpanded = MatrixUtils.getSubMatrix(yExpanded, 0, 0, 0, -1);
//        double cost = cost(xWithBias, thetas, yExpanded, lambda, null, false, softmax);
//        DenseMatrix[] grads = getGradients(xWithBias, thetas, yExpanded, lambda, null, softmax);
//        for (int t = 0; t < thetas.length; t++) {
//            DenseMatrix theta = thetas[t];
//            DenseMatrix grad = grads[t];
//            for (int row = 0; row < theta.numRows(); row++) {
//                for (int col = 0; col < theta.numColumns(); col++) {
//                    double value = theta.get(row, col);
//                    double delta = value*1.0E-5;
//                    theta.set(row, col, value + delta);
//                    double cost2 = cost(xWithBias, thetas, yExpanded, lambda, null, false, softmax);
//                    double gradDiff = (cost2-cost)/delta - grad.get(row, col);
//                    if (Math.abs(gradDiff) > Math.abs(grad.get(row, col)/100.0)) {
//                        System.out.println("    " + gradDiff + "    " + (cost2-cost)/delta + "    " + grad.get(row, col) + "    (" + value + ")");
//                    } else {
//                        System.out.println(gradDiff + "    " + (cost2-cost)/delta + "    " + grad.get(row, col) + "    (" + value + ")");
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
