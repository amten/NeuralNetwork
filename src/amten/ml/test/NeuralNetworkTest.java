package amten.ml.test;

import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixElement;
import amten.ml.matrix.MatrixUtils;

/**
 * Created by Johannes Amtén on 2014-02-24.
 *
 */
public class NeuralNetworkTest {
    public static void main(String[] args) throws Exception {

//        ///////////////////////////////// Gradient Test
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
//        System.exit(0);


        // Use digit image recognition example.
        Matrix x = MatrixUtils.readCSV("C:\\Users\\glenn\\Documents\\Octave\\ML\\ex4\\x.csv", ',', 0);
        Matrix y = MatrixUtils.readCSV("C:\\Users\\glenn\\Documents\\Octave\\ML\\ex4\\y.csv", ',', 0);

//        DenseMatrix x = MatrixUtils.readCSV("C:\\Users\\glenn\\Documents\\Kaggle\\Digit Recognizer\\train_sample_1000.csv", ',', 1);
//        DenseVector y = MatrixUtils.getSubVector(x, 0, -1, x.numColumns()-1);
//        x = MatrixUtils.getSubMatrix(x, 0, -1, 0, x.numColumns()-2);
//        double[] averages = MatrixUtils.getAverages(x);
//        double[] stdDevs = MatrixUtils.getStandardDeviations(x);
//        MatrixUtils.normalizeData(x, averages, stdDevs);


        // Change answers from nominal to 10 booleans
        Matrix yExpanded = new Matrix(x.numRows(), 10);
        for (MatrixElement me: yExpanded)
        {
            me.set(me.col() + 1 ==  y.get(me.row(), 0) ? 1 : 0  );
//            me.set(me.column() ==  y.get(me.row()) ? 1 : 0  );

        }


        int[] hiddenUnits = { 100 };
        double lambda = 1E-8;
        double alpha = 0;
        int batchSize = 100;
        int iterations = 200;
        int threads = 0;
        double inputLayerDropoutRate = 0.2;
        double hiddenLayersDropoutRate = 0.5;
        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork();
        nn.train(x, yExpanded, hiddenUnits, lambda, alpha, batchSize, iterations, threads, inputLayerDropoutRate, hiddenLayersDropoutRate, true, true);

        Matrix h = nn.getPredictions(x);
        int correct = 0;
        for (int i = 0; i < h.numRows(); i++) {
            int prediction = 0;
            double predMaxValue = Double.MIN_VALUE;
            for (int j = 0; j < h.numColumns(); j++) {
                if (h.get(i, j) > predMaxValue) {
                    predMaxValue = h.get(i, j);
                    prediction = j + 1;
//                    prediction = j;
                }
            }
            if (prediction == y.get(i, 0)) {
                correct++;
            }
        }
        System.out.println("Time: " + (System.currentTimeMillis()-startTime)/1000.0 + "s");
        System.out.println("Accuracy: " + (double) correct/h.numRows()*100 + "%");

        System.exit(0);
    }
}
