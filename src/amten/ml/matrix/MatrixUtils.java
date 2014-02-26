package amten.ml.matrix;

import au.com.bytecode.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


/**
 * Created by Johannes Amtén on 2014-02-24.
 *
 */

public class MatrixUtils {

	public static Matrix readCSV(String filename, char separator, int headerLines) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		CSVReader cr = new CSVReader(br, separator, '\"', '\\', headerLines);
		List<String[]> values = cr.readAll();
		cr.close();
		br.close();
		
		int numRows = values.size();
		int numCols = values.get(0).length;
		Matrix m = new Matrix(numRows, numCols);
		for (int row = 0; row < numRows; row++) {
			String[] rowValues = values.get(row);
			for (int col = 0; col < numCols; col++) {
				Double v = Double.parseDouble(rowValues[col]);
				m.set(row, col, v);
			}
		}
		return m;
	}

    public static Matrix random(int rows, int cols) {
        // Create own random generator instead of making calls to Math.random from each thread, which would block each other.
        Random rnd = new Random();
        Matrix m = new Matrix(rows, cols);
        for (MatrixElement me:m) {
            me.set(rnd.nextDouble());
        }
        return m;
    }

	public static Matrix addBiasColumn(Matrix m) {
        Matrix bias = new Matrix(m.numRows(), 1);
        bias.fill(1.0);
        return bias.addColumns(m);
	}

//	/*
//	 * NOTE: If you normalize the input, normalize ONLY on the training data, NOT on the whole data set!
//	 */
//	public static Matrix[] split(Matrix m, float crossValidationPercent, float testPercent)
//	{
//		ArrayList<Integer> rowIndexes = new ArrayList<>();
//		for (int ri = 0; ri < m.numRows(); ri++) {
//			rowIndexes.add(ri);
//		}
//		Collections.shuffle(rowIndexes);
//
//		int numCVRows = Math.round(m.numRows()*crossValidationPercent/100);
//		int numTestRows = Math.round(m.numRows()*testPercent/100);
//		int numTrainRows = m.numRows() - numCVRows - numTestRows;
//
//		Matrix trainMatrix = new Matrix(numTrainRows, m.numColumns());
//		Matrix cvMatrix = new Matrix(numCVRows, m.numColumns());
//		Matrix testMatrix = new Matrix(numTestRows, m.numColumns());
//
//		Iterator<Integer> mRowsIter = rowIndexes.iterator();
//
//		for (int row = 0; row < trainMatrix.numRows(); row++) {
//			int mRow = mRowsIter.next().intValue();
//			for (int col = 0; col < trainMatrix.numColumns(); col++)
//			{
//				double value = m.get(mRow, col);
//				trainMatrix.set(row, col, value);
//			}
//		}
//
//		for (int row = 0; row < cvMatrix.numRows(); row++) {
//			int mRow = mRowsIter.next().intValue();
//			for (int col = 0; col < cvMatrix.numColumns(); col++)
//			{
//				double value = m.get(mRow, col);
//				cvMatrix.set(row, col, value);
//			}
//		}
//
//		for (int row = 0; row < testMatrix.numRows(); row++) {
//			int mRow = mRowsIter.next().intValue();
//			for (int col = 0; col < testMatrix.numColumns(); col++)
//			{
//				double value = m.get(mRow, col);
//				testMatrix.set(row, col, value);
//			}
//		}
//
//		return new Matrix[] {trainMatrix, cvMatrix, testMatrix};
//	}

    public static void split(Matrix x, Matrix y, int batchSize, List<Matrix> batchesX, List<Matrix> batchesY)
    {
        boolean createMatrices = batchesX.size() == 0;
        ArrayList<Integer> rowIndexes = new ArrayList<>();
        for (int ri = 0; ri < x.numRows(); ri++) {
            rowIndexes.add(ri);
        }
        Collections.shuffle(rowIndexes);

        int batchNr = 0;
        Matrix batchX = createMatrices ? new Matrix(Math.min(batchSize, x.numRows()), x.numColumns()) : batchesX.get(batchNr);
        Matrix batchY = createMatrices ? new Matrix(Math.min(batchSize, y.numRows()), y.numColumns()) : batchesY.get(batchNr);
        for (int ri = 0; ri < rowIndexes.size(); ri++) {
            int row = rowIndexes.get(ri);
            for (int col = 0; col < x.numColumns(); col++) {
                double value = x.get(row, col);
                batchX.set(ri % batchSize, col, value);
            }
            for (int col = 0; col < y.numColumns(); col++) {
                double value = y.get(row, col);
                batchY.set(ri % batchSize, col, value);
            }
            int rowsLeft = rowIndexes.size()-ri-1;
            if ((ri + 1) % batchSize == 0 || rowsLeft == 0) {
                if (createMatrices) {
                    batchesX.add(batchX);
                    batchesY.add(batchY);
                    if (rowsLeft > 0) {
                        batchX = new Matrix(Math.min(batchSize, rowsLeft), x.numColumns());
                        batchY = new Matrix(Math.min(batchSize, rowsLeft), y.numColumns());
                    }
                } else if (rowsLeft > 0) {
                    batchNr++;
                    batchX = batchesX.get(batchNr);
                    batchY = batchesY.get(batchNr);
                }
            }
        }
    }

	public static double sigmoid(double x) {
		return 1/(1+Math.exp(-x));
	}


	public static Matrix sigmoid(Matrix m) {
		for (MatrixElement me: m) {
			me.set(sigmoid(me.value()));
		}
		return m;
	}

    public static Matrix softmax(Matrix m) {
        // Subtracting the max value from each value before taking the exponential.
        // This is a trick for preventing overflow.
        // http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
        for (int row = 0 ; row < m.numRows() ; row++) {
            // Find max value
            double max = 0.0;
            for (int col = 0; col < m.numColumns() ; col++) {
                double value = m.get(row, col);
                if (value > max) {
                    max = value;
                }
            }
            // Take exponential of each element and also keep sum of all elements.
            double sum = 0.0;
            for (int col = 0; col < m.numColumns() ; col++) {
                double value = m.get(row, col);
                value -= max;
                value = Math.exp(value);
                m.set(row, col, value);
                sum += value;
            }
            // Divide all elements by the sum
            for (int col = 0; col < m.numColumns() ; col++) {
                m.set(row, col, m.get(row, col)/sum);
            }
        }
        return m;
    }


	public static Matrix sigmoidGradient(Matrix m) {
		// sigmoid(m).*(1-sigmoid(m))
		Matrix t1 = sigmoid(m.copy());
		Matrix t2 = t1.copy();
		t2.scale(-1);
		t2.add(1);
		return t1.multElements(t2);
	}

    public static Matrix rectify(Matrix m) {
        for (MatrixElement me: m) {
            double value = me.value();
            value = Math.max(0, value);
            me.set(value);
        }
        return m;
    }

    public static Matrix rectifyGradient(Matrix m) {
        Matrix gradient = new Matrix(m.numRows(), m.numColumns());
        for (MatrixElement me: m) {
            double g = me.value() >= 0 ? 1 : 0;
            gradient.set(me.row(), me.col(), g);
        }
        return gradient;
    }

	public static Matrix log(Matrix m) {
		for (MatrixElement me: m) {
			me.set(Math.log(me.value()));
		}
		return m;
	}

	public static double[] getAverages(Matrix m) {
		double[] answer = new double[m.numColumns()];
		
		for (int col = 0; col < answer.length; col++) {
            answer[col] = 0;
            for (int row = 0; row < m.numRows() ; row++) {
    			answer[col] += m.get(row, col);
            }
            answer[col] = answer[col] / m.numRows();
		}
		
		return answer;
	}
	

	public static double[] getStandardDeviations(Matrix m) {
		double[] answer = new double[m.numColumns()];
		
		for (int col = 0; col < answer.length; col++) {
			double largestValue = Double.NEGATIVE_INFINITY;
			double smallestValue = Double.POSITIVE_INFINITY;
			for (int row = 0; row < m.numRows(); row++) {
				double value = m.get(row, col);
				if (value > largestValue) {
					largestValue = value;
				}
				if (value < smallestValue) {
					smallestValue = value;
				}
			}
			answer[col] = largestValue - smallestValue;
		}
		
		return answer;
	}
	
	public static void normalizeData(double[] x, double[] averages, double[] standardDeviations) {
		for (int col = 0; col < x.length; col++) {
            double avg = averages[col];
            // Avoid division by zero if standard deviation is zero.
            double stdev = standardDeviations[col] > 0.0 ? standardDeviations[col] : 1.0;
			x[col] = (x[col]-avg)/stdev;
		}
	}
	
	public static void normalizeData(Matrix x, double[] averages, double[] standardDeviations) {
		for (int col = 0; col < x.numColumns(); col++) {
            double avg = averages[col];
            // Avoid division by zero if standard deviation is zero.
            double stdev = standardDeviations[col] > 0.0 ? standardDeviations[col] : 1.0;
			for (int row = 0; row < x.numRows(); row++)	{
				x.set(row, col, (x.get(row, col)-avg)/stdev);
			}
		}
	}

}
