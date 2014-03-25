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

    private Matrix[] myThetas = null;
    private boolean mySoftmax = false;
    private double myInputLayerDropoutRate = 0.0;
    private double myHiddenLayersDropoutRate = 0.0;
    private NNLayerParams[] myLayerParams = null;
    private int myInputWidth = 0;
    private int myInputHeight = 0;


    private double[] myAverages = null;
    private double[] myStdDevs = null;
    private int[] myNumCategories = null;
    private int myNumClasses = 0;

    private transient final ReentrantReadWriteLock myThetasLock = new ReentrantReadWriteLock();
    private transient ExecutorService myExecutorService;

    /**
     * Create an empty Neural Network.
     * Use train() to generate weights.
     */
    public NeuralNetwork() {
    }

    /**
     * Train neural network
     * @param x Training data, input.
     *          One row for each training example.
     *          One column for each attribute.
     * @param numCategories Number of categories for each nominal attribute in x.
     *                      This array should have the same length as the number of columns in x.
     *                      For each nominal attribute, the value should be equal to the number of categories of that attribute.
     *                      For each numeric attribute, the value should be 1.
     *                      If numCategories is null, all attributes in x will be interpreted as numeric.
     * @param y Training data, correct output.
     * @param numClasses Number of classes, if classification.
     *                   1 if regression.
     * @param inputWidth Width of the input.
     *                   For convolution, we have to know the width/height of the input images. (Since they come flattened.)
     *                   If inputWidth is set to zero, a quadratic image with equal width and height is presumed.
                         If you are not using convolutions, this parameter is ignored.
     * @param hiddenLayerParams Number of units in each hidden layer.
     *                          Also, patch-size and pooling-size for convolutional layers.
     * @param weightPenalty L1 weight penalty.
     *                      Even when using dropout, it may be a good idea to have a small weight penalty, to keep weights down and avoid overflow.
     * @param learningRate Initial learning rate.
     *                     If 0, different learning rates will be tried to automatically find a good initial rate.
     * @param batchSize Number of examples to use in each mini-batch.
     * @param iterations Number of iterations (epochs) of training to perform.
     *                   Training may be halted earlier by the user, if debug flag is set.
     * @param threads Number of concurrent calculations.
     *                If 0, threads will automatically be set to the number of CPU cores found.
     * @param inputLayerDropoutRate Dropout rate of input layer.
     *                              Typically set somewhat lower than dropout rate in hidden layer, like 0.2.
     * @param hiddenLayersDropoutRate Dropout rate of hidden layer.
     *                              Typically set somewhat higher than dropout rate in input layer, like 0.5.
     * @param debug If true, training progress will be output to the console.
     *              Also, the user will be able to halt training by pressing enter in the console.
     * @param normalizeNumericData If true, will normalize the data in all numeric columns, by subtracting average and dividing by standard deviation.
     * @throws Exception
     */
    public void train(Matrix x, int[] numCategories, Matrix y, int numClasses, int inputWidth, NNLayerParams[] hiddenLayerParams, double weightPenalty, double learningRate, int batchSize, int iterations,
                      int threads, double inputLayerDropoutRate, double hiddenLayersDropoutRate, boolean debug, boolean normalizeNumericData) throws Exception {
        myNumCategories = numCategories;
        if (myNumCategories == null) {
            myNumCategories = new int[x.numColumns()];
            Arrays.fill(myNumCategories, 1);
        }
        myNumClasses = numClasses;
        mySoftmax = myNumClasses > 1;

        // Add null params for layer 0, which is just the input and last layer, which is just output.
        myLayerParams = new NNLayerParams[hiddenLayerParams.length+2];
        System.arraycopy(hiddenLayerParams, 0, myLayerParams, 1, hiddenLayerParams.length);
        myLayerParams[myLayerParams.length-1] = new NNLayerParams(myNumClasses);

        if (normalizeNumericData) {
            x = x.copy();
            myAverages = new double[x.numColumns()];
            myStdDevs = new double[x.numColumns()];
            for (int col = 0; col < x.numColumns(); col++) {
                if (myNumCategories[col] <= 1) {
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
        x = MatrixUtils.expandNominalAttributes(x, myNumCategories);
        y = MatrixUtils.expandNominalAttributes(y, new int[] {myNumClasses} );

        int inputSize = x.numColumns();
        if (myLayerParams[1].isConvolutional()) {
            // Convolutional network. Save width/height of input images.
            if (inputWidth == 0) {
                // Assume input image has equal width and height.
                myInputWidth = (int) Math.sqrt(inputSize);
                myInputHeight = (int) Math.sqrt(inputSize);
            } else {
                myInputWidth = inputWidth;
                myInputHeight = inputSize / inputWidth;
            }
        } else {
            // Non-convolutional network. Input only has one dimension (width).
            myInputWidth = inputSize;
        }

        initThetas();
        myInputLayerDropoutRate = inputLayerDropoutRate;
        myHiddenLayersDropoutRate = hiddenLayersDropoutRate;
        // If threads == 0, use the same number of threads as cores.
        threads = threads > 0 ? threads : Runtime.getRuntime().availableProcessors();
        myExecutorService = Executors.newFixedThreadPool(threads);

        Reader keyboardReader = System.console() != null ? System.console().reader() : null;
        boolean halted = false;

        List<Matrix> batchesX = new ArrayList<>();
        List<Matrix> batchesY = new ArrayList<>();
        MatrixUtils.split(x, y, batchSize, batchesX, batchesY);
        if (learningRate == 0.0) {
            // Auto-find initial learningRate.
            learningRate = findInitialLearningRate(x, y, batchSize, weightPenalty, debug);
        }

        double cost = getCostThreaded(batchesX, batchesY, weightPenalty);
        LinkedList<Double> oldCosts = new LinkedList<>();
        if (debug) {
            System.out.println("\n\n*** Training network. Press <enter> to halt. ***\n");
            System.out.println("Iteration: 0" + ", Cost: " + String.format("%.3E", cost) + ", Learning rate: " + String.format("%.1E", learningRate));
        }
        for (int i = 0; i < iterations && !halted; i++) {
            // Regenerate the batches each iteration, to get random samples each time.
            MatrixUtils.split(x, y, batchSize, batchesX, batchesY);
            trainOneIterationThreaded(batchesX, batchesY, learningRate, weightPenalty);
            cost = getCostThreaded(batchesX, batchesY, weightPenalty);

            if (oldCosts.size() == 5) {
                // Lower learning rate if cost haven't decreased for 5 iterations.
                double oldCost = oldCosts.remove();
                double minCost = Math.min(cost, Collections.min(oldCosts));
                if (minCost >= oldCost) {
                    learningRate = learningRate*0.1;
                    oldCosts.clear();
                }
            }

            if (debug) {
                System.out.println("Iteration: " + (i + 1) + ", Cost: " + String.format("%.3E", cost) + ", Learning rate: " + String.format("%.1E", learningRate));
            }
            oldCosts.add(cost);

            // Check if user pressed enter in console window to halt computation.
            if (debug && keyboardReader != null && keyboardReader.ready()) {
                while (keyboardReader.ready()) {
                    keyboardReader.read();
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
    public Matrix getPredictions(Matrix x) {
        if (myAverages != null) {
            x = x.copy();
            for (int col = 0; col < x.numColumns(); col++) {
                if (myNumCategories[col] <= 1) {
                    // Normalize numeric column.
                    MatrixUtils.normalizeData(x, col, myAverages[col], myStdDevs[col]);
                }
            }
        }
        // Expand nominal values to groups of booleans.
        x = MatrixUtils.expandNominalAttributes(x, myNumCategories);

        FeedForwardResult[] res = feedForward(x, null);
        return res[res.length-1].output;
    }

    /**
     * Get classification predictions for a number of input examples.
     *
     * @param x Matrix with one row for each input example and one column for each input attribute.
     * @return Matrix with one row for each example and one column containing the predicted class.
     */
    public int[] getPredictedClasses(Matrix x) {
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
//        ArrayList<Integer> numNodes = new ArrayList<>();
//        numNodes.add(inputs);
//        for (int hiddenNodes:hidden) {
//            numNodes.add(hiddenNodes);
//        }
//        numNodes.add(outputs);
//
//        ArrayList<Matrix> thetas = new ArrayList<>();
//        for (int i = 0; i < numNodes.size()-1 ; i++) {
//            int layerInputs = numNodes.get(i) + 1;
//            int layerOutputs = numNodes.get(i+1);
//            thetas.add(createTheta(layerOutputs, layerInputs));
//        }

        ArrayList<Matrix> thetas = new ArrayList<>();

        int numLayers = myLayerParams.length;

        // Input layer has no theta.
        thetas.add(null);

        // Hidden layers
        for (int layer = 1; layer < numLayers; layer++) {
            if (myLayerParams[layer].isConvolutional()) {
                int previousLayerNumFeatureMaps = layer > 1 ? myLayerParams[layer-1].numFeatures : 1;
                int numFeatureMaps = myLayerParams[layer].numFeatures;
                int patchSize = myLayerParams[layer].patchWidth* myLayerParams[layer].patchHeight;
                thetas.add(createTheta(numFeatureMaps, previousLayerNumFeatureMaps*patchSize + 1));
            } else {
                // Meeeeh, not quite. NumFeatureMaps också.
                // Någonting är off här, dock. Numfeaturemaps i non-conv-lager är = antalet noder.
                // Då borde bredden på lagret vara 1 ? Jupp.
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
        int numExamples = x.numRows();
        int numLayers = myLayerParams.length;

//        // Activation layers
//        Matrix[] a = new Matrix[numLayers];
//		a[0] =  x.copy();
//        if (dropoutMasks != null) {
//            a[0].multElements(dropoutMasks[0], a[0]);
//        } else if (myInputLayerDropoutRate > 0.0) {
//            a[0].scale(1.0-myInputLayerDropoutRate);
//        }
//        a[0] = MatrixUtils.addBiasColumn(a[0]);
//        for (int layer = 1; layer < numLayers ; layer++) {
//            // a[layer] = rectify(a[layer-1]*Theta[layer-1]');
//            // ? Store matrices so you don't have to recreate them each time?
//            // Meh... Creating new matrices doesn't seem too bad for performance when tested.
//            a[layer] = new Matrix(numExamples, myThetas[layer-1].numRows());
//            a[layer-1].trans2mult(myThetas[layer-1],  a[layer]);
//            if (layer < numLayers-1) {
//                MatrixUtils.rectify(a[layer]);
//                if (dropoutMasks != null) {
//                    // Dropout hidden nodes, if performing training with dropout.
//                    a[layer].multElements(dropoutMasks[layer], a[layer]);
//                } else if (myHiddenLayersDropoutRate > 0.0) {
//                    // Adjust values if training was done with dropout.
//                    a[layer].scale(1.0-myHiddenLayersDropoutRate);
//                }
//                a[layer] = MatrixUtils.addBiasColumn(a[layer]);
//            } else if (mySoftmax) {
//                MatrixUtils.softmax(a[layer]);
//            }
//        }

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

                ffr[layer].input = layer == 1 ? Convolutions.generatePatchesFromInputLayer(ffr[layer - 1].output, getWidth(layer - 1), patchWidth, patchHeight) :
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
                    double dropoutRate = layer-1 == 0 ? myInputLayerDropoutRate : myHiddenLayersDropoutRate;
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
                if (layer == numLayers-2) {
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
                if ((l == 0 && myInputLayerDropoutRate > 0) || (l > 0 && myHiddenLayersDropoutRate > 0)) {
                    int numColumns = l == 0 ? getWidth(0) : myLayerParams[l].numFeatures*getWidth(l)*getHeight(l);
                    Matrix mask = new Matrix(numExamples, numColumns);
                    double dropoutRate = l == 0 ? myInputLayerDropoutRate : myHiddenLayersDropoutRate;
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
                    boolean useDropout = myInputLayerDropoutRate > 0.0 || myHiddenLayersDropoutRate > 0.0;
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

        int width = myInputWidth;
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
