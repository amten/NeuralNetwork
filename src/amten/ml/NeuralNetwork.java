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
 * Created by Johannes Amtén on 2014-02-24.
 *
 */

public class NeuralNetwork implements Serializable{

    private Matrix[] myThetas = null;
    private boolean mySoftmax = true;
    private double myInputLayerDropoutRate = 0.0;
    private double myHiddenLayersDropoutRate = 0.0;

    private transient final ReentrantReadWriteLock myThetasLock = new ReentrantReadWriteLock();
    private transient ExecutorService myExecutorService;

    public NeuralNetwork() {
    }

    public void train(Matrix x, Matrix y, int[] hiddenUnits, double lambda, double alpha, int batchSize, int iterations,
                      int threads, double inputLayerDropoutRate, double hiddenLayersDropoutRate, boolean softmax, boolean debug) throws Exception {
        initThetas(x.numColumns(), hiddenUnits, y.numColumns());
        mySoftmax = softmax;
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
        if (alpha == 0.0) {
            // Auto-find initial alpha.
            alpha = findInitialAlpha(x, y, lambda, debug);
        }

        double cost = getCostThreaded(batchesX, batchesY, lambda);
        LinkedList<Double> oldCosts = new LinkedList<>();
        if (debug) {
            System.out.println("\n\n*** Training network. Press <enter> to halt. ***\n");
            System.out.println("Iteration: 0" + ", Cost: " + String.format("%.3E", cost) + ", Alpha: " + String.format("%.1E", alpha));
        }
        for (int i = 0; i < iterations && !halted; i++) {
            // Regenerate the batches each iteration, to get random samples each time.
            MatrixUtils.split(x, y, batchSize, batchesX, batchesY);
            trainOneIterationThreaded(batchesX, batchesY, alpha, lambda);
            cost = getCostThreaded(batchesX, batchesY, lambda);

            if (oldCosts.size() == 5) {
                // Lower learning rate if cost haven't decreased for 5 iterations.
                double oldCost = oldCosts.remove();
                double minCost = Math.min(cost, Collections.min(oldCosts));
                if (minCost >= oldCost) {
                    alpha = alpha*0.1;
                    oldCosts.clear();
                }
            }

            if (debug) {
                System.out.println("Iteration: " + (i + 1) + ", Cost: " + String.format("%.3E", cost) + ", Alpha: " + String.format("%.1E", alpha));
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

    public Matrix getPredictions(Matrix x) {
        Matrix[] a = feedForward(x, null);
        return a[a.length-1];
    }

    public double[] getPredictions(double[] x) {
        Matrix xMatrix = new Matrix(new double[][]{x});
        Matrix[] a = feedForward(xMatrix, null);
        return a[a.length-1].getData();
    }

    private void initThetas(int inputs, int[] hidden, int outputs) {
        ArrayList<Integer> numNodes = new ArrayList<>();
        numNodes.add(inputs);
        for (int hiddenNodes:hidden) {
            numNodes.add(hiddenNodes);
        }
        numNodes.add(outputs);

        ArrayList<Matrix> thetas = new ArrayList<>();
        for (int i = 0; i < numNodes.size()-1 ; i++) {
            int layerInputs = numNodes.get(i) + 1;
            int layerOutputs = numNodes.get(i+1);
            Matrix theta = MatrixUtils.random(layerOutputs, layerInputs);
            double epsilon = Math.sqrt(6)/Math.sqrt(layerInputs + layerOutputs);
            theta.scale(epsilon*2);
            theta.add(-epsilon);

            // Set the weight of the biases to a small positive value.
            // This prevents rectified linear units to be stuck at a zero gradient from the beginning.
            for (int row = 0 ; row < theta.numRows() ; row++) {
                theta.set(row, 0, epsilon);
            }
            thetas.add(theta);
        }

        myThetas = thetas.toArray(new Matrix[thetas.size()]);
    }

    private Matrix[] feedForward(Matrix x, Matrix[] dropoutMasks) {
        int numExamples = x.numRows();
        int numLayers = myThetas.length+1;

        // Activation layers
        Matrix[] a = new Matrix[numLayers];
		a[0] =  x.copy();
        if (dropoutMasks != null) {
            a[0].multElements(dropoutMasks[0], a[0]);
        } else if (myInputLayerDropoutRate > 0.0) {
            a[0].scale(1.0-myInputLayerDropoutRate);
        }
        a[0] = MatrixUtils.addBiasColumn(a[0]);
        for (int layer = 1; layer < numLayers ; layer++) {
            // a[layer] = rectify(a[layer-1]*Theta[layer-1]');
            // ? Store matrices so you don't have to recreate them each time?
            // Meh... Creating new matrices doesn't seem too bad for performance when tested.
            a[layer] = new Matrix(numExamples, myThetas[layer-1].numRows());
            a[layer-1].trans2mult(myThetas[layer-1],  a[layer]);
            if (layer < numLayers-1) {
                MatrixUtils.rectify(a[layer]);
                if (dropoutMasks != null) {
                    // Dropout hidden nodes, if performing training with dropout.
                    a[layer].multElements(dropoutMasks[layer], a[layer]);
                } else if (myHiddenLayersDropoutRate > 0.0) {
                    // Adjust values if training was done with dropout.
                    a[layer].scale(1.0-myHiddenLayersDropoutRate);
                }
                a[layer] = MatrixUtils.addBiasColumn(a[layer]);
            } else if (mySoftmax) {
                MatrixUtils.softmax(a[layer]);
            }
        }

		return a;
	}

    private int numberOfNodes() {
        int nodes = 0;
        for (Matrix theta:myThetas) {
            nodes += (theta.numColumns() - 1)* theta.numRows();
        }
        return nodes;
    }

	private double getCost(Matrix x, Matrix y, double lambda, int batchSize) {
        double c = 0.0;

		Matrix[] a = feedForward(x, null);
        Matrix h = a[a.length-1];

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

        if (lambda > 0) {
            // Regularization
            double regSum = 0.0;
            for (Matrix theta:myThetas) {
                for (MatrixElement me: theta.getColumns(1, -1)) {
                    regSum += Math.abs(me.value());
                }
            }
            c += regSum*lambda/numberOfNodes();
        }
	    
	    return c;
	}

    private double getCostThreaded(List<Matrix> batchesX, List<Matrix> batchesY, final double lambda) throws Exception {
        final int batchSize = batchesX.get(0).numRows();
        // Queue up cost calculation in thread pool
        List<Future<Double>> costJobs = new ArrayList<>();
        for (int batchNr = 0 ; batchNr < batchesX.size(); batchNr++) {
            final Matrix bx = batchesX.get(batchNr);
            final Matrix by = batchesY.get(batchNr);
            Callable<Double> costCalculator = new Callable<Double>() {
                public Double call() throws Exception {
                    myThetasLock.readLock().lock();
                    double cost = getCost(bx, by, lambda, batchSize);
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

	private Matrix[] getGradients(Matrix x, Matrix y, double lambda, Matrix[] dropoutMasks, int batchSize) {

        int numLayers = myThetas.length+1;

        // Feed forward and save activations for each layer
        Matrix[] a = feedForward(x, dropoutMasks);

        // Deltas for each layer. (delta for input layer will be left empty)
        Matrix[] delta = new Matrix[numLayers];
        // Start with output layer and then work backwards.
        // delta[numlayers-1] = a[numLayers-1]-y;
        // (delta[numLayers-1] dim is numExamples*output nodes)
        delta[numLayers-1] = a[numLayers-1].copy().add(-1, y);
        for (int layer = numLayers - 2; layer > 0 ; layer--) {
            // delta[layer] = (delta[layer+1]*Theta[layer])(:, 2:end).*rectifyGradient(a[layer-1]*Theta[layer-1]');
            // (delta[layer] dim is examples*nodes in layer)
            delta[layer] = delta[layer+1].mult(myThetas[layer]).getColumns(1, -1);
            delta[layer].multElements(MatrixUtils.rectifyGradient(a[layer-1].trans2mult(myThetas[layer-1])), delta[layer]);
            if (dropoutMasks != null) {
                delta[layer].multElements(dropoutMasks[layer], delta[layer]);
            }
        }

        // Compute gradients for each theta
        Matrix[] thetaGrad = new Matrix[numLayers-1];
        for (int layer = 0 ; layer < numLayers-1 ; layer++) {
            // thetaGrad[layer] = delta[layer+1]'*a[layer];
            // (thetaGrad[layer] dim is nodes in a[layer+1]*nodes in a[layer])
            thetaGrad[layer] = delta[layer+1].trans1mult(a[layer]);
            // Have to use batch size and not number of rows, in case that the last batch contains fewer examples.
            thetaGrad[layer].scale(1.0/batchSize);
        }

        if (lambda > 0) {
            // Add regularization terms
            int numNodes = numberOfNodes();
            for (int thetaNr = 0; thetaNr < numLayers-1 ; thetaNr++) {
                Matrix theta = myThetas[thetaNr];
                Matrix grad = thetaGrad[thetaNr];
                for (int row = 0; row < grad.numRows() ; row++) {
                    for (int col = 1; col < grad.numColumns(); col++) {
                        double regTerm = lambda/numNodes*Math.signum(theta.get(row, col));
                        grad.add(row, col, regTerm);
                    }
                }
            }
        }

		return thetaGrad;
	}

    private Matrix[] generateDropoutMasks(int examples) {
        // Create own random generator instead of making calls to Math.random from each thread, which would block each other.
        Random rnd = new Random();

        Matrix[] masks = new Matrix[myThetas.length];
        for (int i = 0; i < masks.length ; i++) {
            Matrix mask = new Matrix(examples, myThetas[i].numColumns()-1);
            double dropoutRate = i == 0 ? myInputLayerDropoutRate : myHiddenLayersDropoutRate;
            for (MatrixElement me : mask) {
                me.set(rnd.nextDouble() > dropoutRate ? 1.0 : 0.0);
            }
            masks[i] = mask;
        }
        return masks;
    }
	
    private void trainOneIterationThreaded(List<Matrix> batchesX, List<Matrix> batchesY, final double alpha, final double lambda) throws Exception {
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
                    Matrix[] gradients = getGradients(bx, by, lambda, dropoutMasks, batchSize);
                    myThetasLock.readLock().unlock();
                    myThetasLock.writeLock().lock();
                    for (int theta = 0; theta < myThetas.length; theta++) {
                        myThetas[theta].add(-alpha, gradients[theta]);
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
            res[i] = ms[i].copy();
        }
        return res;
    }

    private double findInitialAlpha(Matrix x, Matrix y, double lambda, boolean debug) throws Exception {
        int numUsedTrainingExamples = 5000;
        int batchSize = 100;
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
        double alpha = 1.0E-10;
        trainOneIterationThreaded(batchesX, batchesY, alpha, lambda);
        double cost = getCostThreaded(batchesX, batchesY, lambda);
        if (debug) {
            System.out.println("\n\nAuto-finding learning rate, alpha");
            System.out.println("Alpha: " + String.format("%.1E", alpha) + " Cost: " + cost); ////////////////////////////
        }
        myThetas = deepCopy(startThetas);
        double lastCost = Double.MAX_VALUE;
        double lastAlpha = alpha;
        while (cost < lastCost) {
            lastCost = cost;
            lastAlpha = alpha;
            alpha = alpha*10.0;
            trainOneIterationThreaded(batchesX, batchesY, alpha, lambda);
            cost = getCostThreaded(batchesX, batchesY, lambda);
            if (debug) {
                System.out.println("Alpha: " + String.format("%.1E", alpha) + " Cost: " + cost); ////////////////////////////
            }
            myThetas = deepCopy(startThetas);
        }

        if (debug) {
            System.out.println("Using alpha: " + String.format("%.1E", lastAlpha));
        }
        return lastAlpha;
    }


}
