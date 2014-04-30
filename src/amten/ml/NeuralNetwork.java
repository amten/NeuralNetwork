package amten.ml;

import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixElement;
import amten.ml.matrix.MatrixUtils;

import java.io.Reader;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * (Convolutional) Neural network implementation with dropout and rectified linear units.
 * Can perform regression or classification.
 * Training is done by multithreaded mini-batch gradient descent with native matrix lib.
 *
 * @author Johannes Amtén
 *
 */

public class NeuralNetwork implements Serializable{

    private NNParams myParams = null;

    private Matrix[] myThetas = null;
    private boolean mySoftmax = false;
    private int myInputHeight = 0;
    private int myNumOutputs = 0;
    private NNParams.NNLayerParams[] myLayerParams = null;

    private double[] myAverages = null;
    private double[] myStdDevs = null;


    private transient final ReentrantReadWriteLock myThetasLock = new ReentrantReadWriteLock();
    private transient ExecutorService myExecutorService;

    /**
     * Create an empty Neural Network.
     * Use train() to generate weights.
     *
     * @param p The parameters of the neural network.
     */
    public NeuralNetwork(NNParams p) {
        myParams = p;
    }

    /**
     * Train neural network
     * @param x Training data, input.
     *          One row for each training example.
     *          One column for each attribute.
     * @param y Training data, correct output.
     * @throws Exception
     */
    public void train(Matrix x, Matrix y) throws Exception {
        if (myParams.numCategories == null) {
            myParams.numCategories = new int[x.numColumns()];
            Arrays.fill(myParams.numCategories, 1);
        }
        mySoftmax = myParams.numClasses > 1;
        myNumOutputs = myParams.numClasses > 1 ? myParams.numClasses : y.numColumns();

        // Add null params for layer 0, which is just the input and last layer, which is just output.
        myLayerParams = new NNParams.NNLayerParams[myParams.hiddenLayerParams.length+2];
        System.arraycopy(myParams.hiddenLayerParams, 0, myLayerParams, 1, myParams.hiddenLayerParams.length);
        myLayerParams[myLayerParams.length-1] = new NNParams.NNLayerParams(myNumOutputs);

        if (myParams.normalizeNumericData) {
            if (myParams.dataLoader != null) {
                throw new Exception("With load on demand, data must be normalized before being sent to NeuralNetwork.");
            }
            x = x.copy();
            myAverages = new double[x.numColumns()];
            myStdDevs = new double[x.numColumns()];
            for (int col = 0; col < x.numColumns(); col++) {
                if (myParams.numCategories[col] <= 1) {
                    // Normalize numeric column.
                    myAverages[col] = MatrixUtils.getAverage(x, col);
                    myStdDevs[col] = MatrixUtils.getStandardDeviation(x, col);
                    MatrixUtils.normalizeData(x, col, myAverages[col], myStdDevs[col]);
                }
            }
        } else {
            myAverages = null;
            myStdDevs = null;
        }

        // Expand nominal values to groups of booleans.
        x = MatrixUtils.expandNominalAttributes(x, myParams.numCategories);
        if (mySoftmax) {
            y = MatrixUtils.expandNominalAttributes(y, new int[] {myNumOutputs} );
        }

        int inputSize = myParams.dataLoader == null ? x.numColumns() : myParams.dataLoader.getDataSize();

        if (myLayerParams[1].isConvolutional()) {
            // Convolutional network. Save width/height of input images.
            myParams.numInputChannels = myParams.numInputChannels > 0 ? myParams.numInputChannels : 1;
            if (myParams.inputWidth == 0) {
                // Assume input image has equal width and height.
                myParams.inputWidth = (int) Math.sqrt(inputSize/myParams.numInputChannels);
                myInputHeight = (int) Math.sqrt(inputSize/myParams.numInputChannels);
            } else {
                myInputHeight = inputSize / (myParams.numInputChannels*myParams.inputWidth);
            }
        } else {
            // Non-convolutional network. Input only has one dimension (width).
            myParams.inputWidth = inputSize;
        }

        if (myParams.batchSize == 0) {
            // Auto-choose batch-size.
            // 100 for fully connected network and 1 for convolutional network.
            myParams.batchSize = myLayerParams[1].isConvolutional() ? 1 : 100;
        }

        initThetas();
        // If threads == 0, use the same number of threads as cores.
        myParams.numThreads = myParams.numThreads > 0 ? myParams.numThreads : Runtime.getRuntime().availableProcessors();
        myExecutorService = Executors.newFixedThreadPool(myParams.numThreads);

        Reader keyboardReader = System.console() != null ? System.console().reader() : null;
        boolean halted = false;

        List<Matrix> batchesX = new ArrayList<>();
        List<Matrix> batchesY = new ArrayList<>();
        MatrixUtils.split(x, y, myParams.batchSize, batchesX, batchesY);
        if (myParams.learningRate == 0.0) {
            // Auto-find initial learningRate.
            myParams.learningRate = findInitialLearningRate(x, y, myParams.batchSize, myParams.weightPenalty, myParams.debug);
        }

        double cost = getCostThreaded(batchesX, batchesY, myParams.weightPenalty);
        LinkedList<Double> fiveLastCosts = new LinkedList<>();
        LinkedList<Double> tenLastCosts = new LinkedList<>();
        if (myParams.debug) {
            System.out.println("\n\n*** Training network. Press <enter> to halt. ***\n");
            if (mySoftmax) {
                System.out.println("Iteration: 0" + ", Cost: " + String.format("%.3E", cost) + ", Learning rate: " + String.format("%.1E", myParams.learningRate));
            } else {
                System.out.println("Iteration: 0" + ", RMSE: " + String.format("%.3E", Math.sqrt(cost*2.0/ myNumOutputs)) + ", Learning rate: " + String.format("%.1E", myParams.learningRate));
            }
        }
        for (int i = 0; i < myParams.maxIterations && !halted; i++) {
            // Regenerate the batches each iteration, to get random samples each time.
            MatrixUtils.split(x, y, myParams.batchSize, batchesX, batchesY);
            trainOneIterationThreaded(batchesX, batchesY, myParams.learningRate, myParams.weightPenalty);
            cost = getCostThreaded(batchesX, batchesY, myParams.weightPenalty);

            if (fiveLastCosts.size() == 5) {
                // Lower learning rate if cost haven't decreased for 5 iterations.
                double oldCost = fiveLastCosts.remove();
                double minCost = Math.min(cost, Collections.min(fiveLastCosts));
                if (minCost >= oldCost) {
                    myParams.learningRate = myParams.learningRate*0.1;
                    fiveLastCosts.clear();
                }
            }
            if (tenLastCosts.size() == 10) {
                double minCost = Math.min(cost, Collections.min(tenLastCosts));
                double maxCost = Math.max(cost, Collections.max(tenLastCosts));
                tenLastCosts.remove();
                if ((maxCost-minCost)/minCost < myParams.convergenceThreshold) {
                    // If cost hasn't moved by more than the threshold fraction for the last 10 iterations,
                    // we declare convergence and stop training.
                    halted = true;
                }
            }

            if (myParams.debug) {
                if (mySoftmax) {
                    System.out.println("Iteration: " + (i + 1) + ", Cost: " + String.format("%.3E", cost) + ", Learning rate: " + String.format("%.1E", myParams.learningRate));
                } else {
                    System.out.println("Iteration: " + (i + 1) + ", RMSE: " + String.format("%.3E", Math.sqrt(cost*2.0/ myNumOutputs)) + ", Learning rate: " + String.format("%.1E", myParams.learningRate));
                }
            }
            fiveLastCosts.add(cost);
            tenLastCosts.add(cost);

            if (myParams.debug && System.in.available() > 0) {
                while (System.in.available() > 0) {
                    System.in.read();
                }
                System.out.println("Training halted by user.");
                halted = true;
            }
        }

        myExecutorService.shutdown();
    }

    /**
     * Get predictions for a number of input examples.
     *
     * @param x Matrix with one row for each input example and one column for each input attribute.
     * @return Matrix with one row for each example.
     *          If regression, only one column containing the predicted value.
     *          If classification, one column for each class, containing the predicted probability of that class.
     */
    public Matrix getPredictions(Matrix x) throws Exception {
        if (myAverages != null) {
            x = x.copy();
            for (int col = 0; col < x.numColumns(); col++) {
                if (myParams.numCategories[col] <= 1) {
                    // Normalize numeric column.
                    MatrixUtils.normalizeData(x, col, myAverages[col], myStdDevs[col]);
                }
            }
        }
        // Expand nominal values to groups of booleans.
        x = MatrixUtils.expandNominalAttributes(x, myParams.numCategories);

        if (x.numRows() > myParams.batchSize) {
            // Batch and thread calculations.
            final Matrix predictions = new Matrix(x.numRows(), myNumOutputs);
            ExecutorService es = Executors.newFixedThreadPool(myParams.numThreads);

            List<Future> queuedJobs = new ArrayList<>();
            for (int row = 0 ; row < x.numRows(); row += myParams.batchSize) {
                final int startRow = row;
                final int endRow = Math.min(startRow + myParams.batchSize - 1, x.numRows()-1);
                final Matrix batchX = x.getRows(startRow, endRow);

                Runnable predictionsCalculator = new Runnable() {
                    public void run() {
                    FeedForwardResult[] ffRes = feedForward(batchX, null);
                    Matrix batchPredictions = ffRes[ffRes.length-1].output;
                    for (int batchRow = 0; batchRow < batchPredictions.numRows(); batchRow++) {
                        for (int batchCol = 0; batchCol < batchPredictions.numColumns(); batchCol++) {
                            predictions.set(startRow+batchRow, batchCol, batchPredictions.get(batchRow, batchCol));
                        }
                    }
                    }
                };
                queuedJobs.add(es.submit(predictionsCalculator));
            }

            // Wait for all gradient calcs to be done.
            for (Future job:queuedJobs) {
                job.get();
            }
            es.shutdown();

            return predictions;
        } else {
            FeedForwardResult[] res = feedForward(x, null);
            return res[res.length-1].output;
        }
    }

    /**
     * Get classification predictions for a number of input examples.
     *
     * @param x Matrix with one row for each input example and one column for each input attribute.
     * @return Matrix with one row for each example and one column containing the predicted class.
     */
    public int[] getPredictedClasses(Matrix x) throws Exception {
        Matrix y = getPredictions(x);
        int[] predictedClasses = new int[x.numRows()];
        for (int row = 0; row < y.numRows(); row++) {
            int prediction = 0;
            double predMaxValue = -1.0;
            for (int col = 0; col < y.numColumns(); col++) {
                if (y.get(row, col) > predMaxValue) {
                    predMaxValue = y.get(row, col);
                    prediction = col;
                }
            }
            predictedClasses[row] = prediction;
        }
        return predictedClasses;
    }

//    public double[] getPredictions(double[] x) {
//        Matrix xMatrix = new Matrix(new double[][]{x});
//        Matrix[] a = feedForward(xMatrix, null);
//        return a[a.length-1].getData();
//    }


    private Matrix createTheta(int rows, int cols) {
        Matrix theta = MatrixUtils.random(rows, cols);
        double epsilon = Math.sqrt(6)/Math.sqrt(rows + cols);
        theta.scale(epsilon*2);
        theta.add(-epsilon);

        // Set the weight of the biases to a small positive value.
        // This prevents rectified linear units to be stuck at a zero gradient from the beginning.
        for (int row = 0 ; row < theta.numRows() ; row++) {
            theta.set(row, 0, epsilon);
        }
        return theta;
    }

    private void initThetas() {
        ArrayList<Matrix> thetas = new ArrayList<>();

        int numLayers = myLayerParams.length;

        // Input layer has no theta.
        thetas.add(null);

        // Hidden layers
        for (int layer = 1; layer < numLayers; layer++) {
            if (myLayerParams[layer].isConvolutional()) {
                int previousLayerNumFeatureMaps = layer > 1 ? myLayerParams[layer-1].numFeatures : myParams.numInputChannels;
                int numFeatureMaps = myLayerParams[layer].numFeatures;
                int patchSize = myLayerParams[layer].patchWidth* myLayerParams[layer].patchHeight;
                thetas.add(createTheta(numFeatureMaps, previousLayerNumFeatureMaps*patchSize + 1));
            } else {
                int layerInputs;
                if (layer-1 == 0) {
                    layerInputs = getWidth(0) + 1;
                } else if (myLayerParams[layer-1].isConvolutional()) {
                    layerInputs = myLayerParams[layer-1].numFeatures*getWidth(layer-1)*getHeight(layer-1) + 1;
                } else {
                    layerInputs = myLayerParams[layer-1].numFeatures + 1;
                }
                int layerOutputs = myLayerParams[layer].numFeatures;
                thetas.add(createTheta(layerOutputs, layerInputs));
            }
        }

        myThetas = thetas.toArray(new Matrix[thetas.size()]);
    }

    private FeedForwardResult[] feedForward(Matrix x, Matrix[] dropoutMasks) {
        if (myParams.dataLoader != null) {
            // Load data on demand. To large dataset to fit in memory.
            try {
                x = myParams.dataLoader.loadData(x);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        int numExamples = x.numRows();
        int numLayers = myLayerParams.length;

        FeedForwardResult[] ffr = new FeedForwardResult[numLayers];

        ffr[0] = new FeedForwardResult();
        ffr[0].output = x.copy();
        for (int layer = 1; layer < numLayers; layer++) {
            ffr[layer] = new FeedForwardResult();

            if (myLayerParams[layer].isConvolutional()) {
                //Convolutional layer
                int patchWidth = myLayerParams[layer].patchWidth;
                int patchHeight = myLayerParams[layer].patchHeight;
                int poolWidth = myLayerParams[layer].poolWidth;
                int poolHeight = myLayerParams[layer].poolHeight;

                ffr[layer].input = layer == 1 ? Convolutions.generatePatchesFromInputLayer(ffr[layer - 1].output, getWidth(layer - 1), getHeight(layer - 1), patchWidth, patchHeight) :
                        Convolutions.generatePatchesFromHiddenLayer(ffr[layer - 1].output, getWidth(layer - 1), getHeight(layer - 1), patchWidth, patchHeight);
                ffr[layer].input = MatrixUtils.addBiasColumn(ffr[layer].input); // Move into generatePatches() ?

                ffr[layer].output = ffr[layer].input.trans2mult(myThetas[layer]);

                if (myLayerParams[layer].isPooled()) {
                    Convolutions.PoolingResult pr = Convolutions.maxPool(ffr[layer].output, (getWidth(layer - 1) - patchWidth + 1), (getHeight(layer - 1) - patchHeight + 1), poolWidth, poolHeight);
                    ffr[layer].numRowsPrePool = ffr[layer].output.numRows();
                    ffr[layer].output = pr.pooledActivations;
                    ffr[layer].prePoolRowIndexes = pr.prePoolRowIndexes;
                }
            } else {
                // Fully connected layer
                if (layer > 1 && myLayerParams[layer-1].isConvolutional()) {
                    // Reorder output from previous convolutional layer, so that all patches are columns instead of rows.
                    int numPatches = getWidth(layer-1)*getHeight(layer-1);
                    int numFeatureMaps = myLayerParams[layer-1].numFeatures;
                    ffr[layer-1].output = Convolutions.movePatchesToColumns(ffr[layer - 1].output, numExamples, numFeatureMaps, numPatches);
                }
                if (dropoutMasks != null && dropoutMasks[layer-1] != null) {
                    // Dropout hidden nodes, if performing training with dropout.
                    ffr[layer-1].output.multElements(dropoutMasks[layer - 1], ffr[layer - 1].output);
                } else {
                    double dropoutRate = layer-1 == 0 ? myParams.inputLayerDropoutRate : myParams.hiddenLayersDropoutRate;
                    if (dropoutRate > 0.0) {
                        // Adjust values if training was done with dropout.
                        ffr[layer-1].output.scale(1.0 - dropoutRate);
                    }
                }
                ffr[layer].input = MatrixUtils.addBiasColumn(ffr[layer-1].output);
                ffr[layer].output = ffr[layer].input.trans2mult(myThetas[layer]);
            }
            if (layer < numLayers-1) {
                MatrixUtils.rectify(ffr[layer].output);
            } else if (mySoftmax) {
                MatrixUtils.softmax(ffr[layer].output);
            }
        }

        return ffr;
    }

    private int numberOfNodes() {
        int nodes = 0;
        for (int t = 1; t < myThetas.length; t++) {
            nodes += (myThetas[t].numColumns() - 1)* myThetas[t].numRows();
        }
        return nodes;
    }

    private double getCost(Matrix x, Matrix y, double weightPenalty, int batchSize) {
        double c = 0.0;

        FeedForwardResult[] ffr = feedForward(x, null); //Can't use getPredictions because of normalization.
        Matrix h = ffr[ffr.length-1].output;

        if (mySoftmax) {
            for (int row = 0; row < y.numRows(); row++) {
                for (int col = 0; col < y.numColumns(); col++) {
                    if (y.get(row, col) > 0.99) {
                        c -= Math.log(h.get(row, col));
                    }
                }
            }
            // Have to use batch size and not number of rows, in case that the last batch contains fewer examples.
            c = c/batchSize;
        } else {
            // t1=(h-y).*(h-y)
            Matrix t1 = h.copy().add(-1,  y);
            t1.multElements(t1, t1);
            // sum = sum(sum(t1))
            for (MatrixElement me: t1) {
                c += me.value();
            }
            // Have to use batch size and not number of rows, in case that the last batch contains fewer examples.
            c = c/(2*batchSize);
        }

        if (weightPenalty > 0) {
            // Regularization
            double regSum = 0.0;
            for (int t = 1; t < myThetas.length; t++) {
                for (MatrixElement me: myThetas[t].getColumns(1, -1)) {
                    regSum += Math.abs(me.value());
                }
            }
            c += regSum*weightPenalty/numberOfNodes();
        }

        return c;
    }

    private double getCostThreaded(List<Matrix> batchesX, List<Matrix> batchesY, final double weightPenalty) throws Exception {
        final int batchSize = batchesX.get(0).numRows();
        // Queue up cost calculation in thread pool
        List<Future<Double>> costJobs = new ArrayList<>();
        for (int batchNr = 0 ; batchNr < batchesX.size(); batchNr++) {
            final Matrix bx = batchesX.get(batchNr);
            final Matrix by = batchesY.get(batchNr);
            Callable<Double> costCalculator = new Callable<Double>() {
                public Double call() throws Exception {
                    myThetasLock.readLock().lock();
                    double cost = getCost(bx, by, weightPenalty, batchSize);
                    myThetasLock.readLock().unlock();
                    return cost;
                }
            };
            costJobs.add(myExecutorService.submit(costCalculator));
        }

        // Get cost
        double cost = 0;
        for (Future<Double> job:costJobs) {
            cost += job.get();
        }
        cost = cost/batchesX.size();
        return cost;
    }

    private Matrix[] getGradients(Matrix x, Matrix y, double weightPenalty, Matrix[] dropoutMasks, int batchSize) {

        int numLayers = myLayerParams.length;
        int numExamples = x.numRows();

        // Feed forward and save activations for each layer
        FeedForwardResult[] ffr = feedForward(x, dropoutMasks);

        Matrix[] delta = new Matrix[numLayers];
        // Start with output layer and then work backwards.
        // delta[numlayers-1] = a[numLayers-1]-y;
        // (delta[numLayers-1] dim is numExamples*output nodes)
        delta[numLayers-1] = ffr[numLayers-1].output.copy().add(-1, y);

        for (int layer = numLayers-2; layer >=1; layer--) {
            delta[layer] = delta[layer+1].mult(myThetas[layer+1]).getColumns(1, -1);
            if (dropoutMasks != null && dropoutMasks[layer] != null) {
                delta[layer].multElements(dropoutMasks[layer], delta[layer]);
            }
            if (myLayerParams[layer].isConvolutional()) {
                // Convolutional layer.
                int numFeatureMaps = myLayerParams[layer].numFeatures;
                int patchWidth = myLayerParams[layer].patchWidth;
                int patchHeight = myLayerParams[layer].patchHeight;
                if (!myLayerParams[layer+1].isConvolutional()) {
                    int numPatches = getWidth(layer)*getHeight(layer);
                    delta[layer] = Convolutions.movePatchesToRows(delta[layer], numExamples, numFeatureMaps, numPatches);
                } else {
                    delta[layer] = Convolutions.antiPatchDeltas(delta[layer], getWidth(layer), getHeight(layer), patchWidth, patchHeight);
                }
                if (myLayerParams[layer].isPooled()) {
                    delta[layer] = Convolutions.antiPoolDelta(delta[layer], ffr[layer].prePoolRowIndexes, ffr[layer].numRowsPrePool);
                }
            }
            delta[layer].multElements(MatrixUtils.rectifyGradient(ffr[layer].input.trans2mult(myThetas[layer])), delta[layer]);
        }

        // Compute gradients for each theta
        Matrix[] thetaGrad = new Matrix[numLayers];
        for (int layer = 1; layer < numLayers; layer++) {
            // thetaGrad[layer] = delta[layer+1]'*a[layer];
            // (thetaGrad[layer] dim is nodes in a[layer+1]*nodes in a[layer])
            thetaGrad[layer] = delta[layer].trans1mult(ffr[layer].input);
            // Have to use batch size and not number of rows, in case that the last batch contains fewer examples.
            thetaGrad[layer].scale(1.0 / batchSize);
        }

        if (weightPenalty > 0) {
            // Add regularization terms
            int numNodes = numberOfNodes();
            for (int thetaNr = 1; thetaNr < numLayers ; thetaNr++) {
                Matrix theta = myThetas[thetaNr];
                Matrix grad = thetaGrad[thetaNr];
                for (int row = 0; row < grad.numRows() ; row++) {
                    for (int col = 1; col < grad.numColumns(); col++) {
                        double regTerm = weightPenalty/numNodes*Math.signum(theta.get(row, col));
                        grad.add(row, col, regTerm);
                    }
                }
            }
        }

        return thetaGrad;
    }

    private Matrix[] generateDropoutMasks(int numExamples) {
        // Create own random generator instead of making calls to Math.random from each thread, which would block each other.
        Random rnd = new Random();

        int numLayers = myLayerParams.length;
        // Don't dropout output layer.
        Matrix[] masks = new Matrix[numLayers-1];

        for (int l = 0; l < masks.length ; l++) {
            // Don't dropout input to convolutional layers.
            if (!myLayerParams[l+1].isConvolutional()) {
                if ((l == 0 && myParams.inputLayerDropoutRate > 0) || (l > 0 && myParams.hiddenLayersDropoutRate > 0)) {
                    int numColumns = l == 0 ? getWidth(0) : myLayerParams[l].numFeatures*getWidth(l)*getHeight(l);
                    Matrix mask = new Matrix(numExamples, numColumns);
                    double dropoutRate = l == 0 ? myParams.inputLayerDropoutRate : myParams.hiddenLayersDropoutRate;
                    for (MatrixElement me : mask) {
                        me.set(rnd.nextDouble() > dropoutRate ? 1.0 : 0.0);
                    }
                    masks[l] = mask;
                }
            }
        }

        return masks;
    }

    private void trainOneIterationThreaded(List<Matrix> batchesX, List<Matrix> batchesY, final double learningRate, final double weightPenalty) throws Exception {
        final int batchSize = batchesX.get(0).numRows();

        // Queue up all batches for gradient computation in the thread pool.
        List<Future> queuedJobs = new ArrayList<>();
        for (int batchNr = 0 ; batchNr < batchesX.size(); batchNr++) {
            final Matrix bx = batchesX.get(batchNr);
            final Matrix by = batchesY.get(batchNr);

            Runnable batchGradientCalculator = new Runnable() {
                public void run() {
                    boolean useDropout = myParams.inputLayerDropoutRate > 0.0 || myParams.hiddenLayersDropoutRate > 0.0;
                    Matrix[] dropoutMasks = useDropout ? generateDropoutMasks(by.numRows()) : null;
                    myThetasLock.readLock().lock();
                    Matrix[] gradients = getGradients(bx, by, weightPenalty, dropoutMasks, batchSize);
                    myThetasLock.readLock().unlock();
                    myThetasLock.writeLock().lock();
                    for (int theta = 1; theta < myThetas.length; theta++) {
                        myThetas[theta].add(-learningRate, gradients[theta]);
                    }
                    myThetasLock.writeLock().unlock();
                }
            };
            queuedJobs.add(myExecutorService.submit(batchGradientCalculator));
        }

        // Wait for all gradient calcs to be done.
        for (Future job:queuedJobs) {
            job.get();
        }
    }


    private Matrix[] deepCopy(Matrix[] ms) {
        Matrix[] res = new Matrix[ms.length];
        for (int i = 0; i < res.length ; i++) {
            res[i] = ms[i] != null ? ms[i].copy() : null;
        }
        return res;
    }

    private double findInitialLearningRate(Matrix x, Matrix y, int batchSize, double weightPenalty, boolean debug) throws Exception {
        int numUsedTrainingExamples = 5000;
        int numBatches = numUsedTrainingExamples/batchSize;

        List<Matrix> batchesX = new ArrayList<>();
        List<Matrix> batchesY = new ArrayList<>();
        MatrixUtils.split(x, y, batchSize, batchesX, batchesY);
        while (batchesX.size() < numBatches) {
            batchesX.addAll(batchesX);
            batchesY.addAll(batchesY);
        }
        if (batchesX.size() > numBatches) {
            batchesX = batchesX.subList(0, numBatches);
            batchesY = batchesY.subList(0, numBatches);
        }

        Matrix[] startThetas = deepCopy(myThetas);
        double lr = 1.0E-10;
        trainOneIterationThreaded(batchesX, batchesY, lr, weightPenalty);
        double cost = getCostThreaded(batchesX, batchesY, weightPenalty);
        if (debug) {
            System.out.println("\n\nAuto-finding learning rate.");
            System.out.println("Learning rate: " + String.format("%.1E", lr) + " Cost: " + cost);
        }
        myThetas = deepCopy(startThetas);
        double lastCost = Double.MAX_VALUE;
        double lastLR = lr;
        while (cost < lastCost) {
            lastCost = cost;
            lastLR = lr;
            lr = lr*10.0;
            trainOneIterationThreaded(batchesX, batchesY, lr, weightPenalty);
            cost = getCostThreaded(batchesX, batchesY, weightPenalty);
            if (debug) {
                System.out.println("Learning rate: " + String.format("%.1E", lr) + " Cost: " + cost);
            }
            myThetas = deepCopy(startThetas);
        }

        if (debug) {
            System.out.println("Using learning rate: " + String.format("%.1E", lastLR));
        }
        return lastLR;
    }

    private int getWidth(int layer) {
        if (layer > 0 && !myLayerParams[layer].isConvolutional()) {
            return 1;
        }

        int width = myParams.inputWidth;
        for (int l = 1; l <= layer; l++) {
            // Calc number of patch-rows next layer will have. Convolution and pooling.
            int patchWidth = myLayerParams[l].patchWidth;
            int poolWidth = myLayerParams[l].poolWidth;
            width = width-patchWidth+1;
            width = width%poolWidth == 0 ? width/poolWidth :
                                            width/poolWidth + 1;
        }
        return width;
    }

    private int getHeight(int layer) {
        if (layer > 0 && !myLayerParams[layer].isConvolutional()) {
            return 1;
        }

        int height = myInputHeight;
        for (int l = 1; l <= layer; l++) {
            // Calc number of patch-rows next layer will have. Convolution and pooling.
            int patchHeight = myLayerParams[l].patchHeight;
            int poolHeight = myLayerParams[l].poolHeight;
            height = height-patchHeight+1;
            height = height%poolHeight == 0 ? height/poolHeight :
                                                height/poolHeight + 1;
        }
        return height;
    }

    private static class FeedForwardResult {
        public Matrix input = null;
        public Matrix output = null;
        public Matrix prePoolRowIndexes = null;
        public int numRowsPrePool = 0;
    }

}
