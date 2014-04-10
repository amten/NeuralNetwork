package amten.ml.examples;

import amten.ml.NNParams;
import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixUtils;

/**
 * Examples of using NeuralNetwork for regression.
 *
 * @author Johannes Amtén
 */
public class NNRegressionExample {

    /**
     * Performs regression on a dataset of car prices for cars with different features.
     * <br></br>
     * Uses file /example_data/Car_Prices.csv
     */
    public static void runCarPricesRegression() throws Exception {
        System.out.println("Running regression on Car Prices dataset...\n");
        // Read data from CSV-file
        int headerRows = 1;
        char separator = ',';
        Matrix data = MatrixUtils.readCSV("example_data/Car_Prices.csv", separator, headerRows);

        // Split data into training set and crossvalidation set.
        float crossValidationPercent = 33;
        Matrix[] split = MatrixUtils.split(data, crossValidationPercent, 0);
        Matrix dataTrain = split[0];
        Matrix dataCV = split[1];

        // 15:th column contains the correct price. The rest are the indata.
        Matrix xTrain = dataTrain.getColumns(0, 13);
        Matrix yTrain = dataTrain.getColumns(14, 14);
        Matrix xCV = dataCV.getColumns(0, 13);
        Matrix yCV = dataCV.getColumns(14, 14);

        // Use default parameters; single hidden layer with 100 units.
        NNParams params = new NNParams();

        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork(params);
        nn.train(xTrain, yTrain);
        System.out.println("\nTraining time: " + String.format("%.3g", (System.currentTimeMillis() - startTime) / 1000.0) + "s");

        Matrix predictions = nn.getPredictions(xTrain);
        double error = 0;
        for (int i = 0; i < predictions.numRows(); i++) {
            error += Math.pow(predictions.get(i, 0) - yTrain.get(i, 0), 2);
        }
        error = Math.sqrt(error / predictions.numRows());
        System.out.println("Training set root mean squared error: " + String.format("%.4g", error));

        predictions = nn.getPredictions(xCV);
        error = 0;
        for (int i = 0; i < predictions.numRows(); i++) {
            error += Math.pow(predictions.get(i, 0) - yCV.get(i, 0), 2);
        }
        error = Math.sqrt(error / predictions.numRows());
        System.out.println("Crossvalidation set root mean squared error: " + String.format("%.4g", error));
    }


    public static void main(String[] args) throws Exception {
        runCarPricesRegression();
    }
}
