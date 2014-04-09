package amten.ml.matrix;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

import java.io.*;
import java.util.*;


/**
 * Some helper methods for manipulating matrices.
 *
 * @author Johannes Amtén
 *
 */

public class MatrixUtils {

    /**
     * Reads a CSV-file from disk into a Matrix.
     *
     * @param filename
     * @param separator Separator character between values.
     * @param headerLines Number of header lines to skip before reading data.
     * @return Matrix
     * @throws IOException
     */
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

    public static void writeCSV(Matrix m, String filename) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
        CSVWriter cw = new CSVWriter(bw, ',', CSVWriter.NO_QUOTE_CHARACTER);

        ArrayList<String[]> rows = new ArrayList<>();
        for (int row = 0; row < m.numRows(); row++) {
            String[] rowValues = new String[m.numColumns()];
            for (int col = 0; col < m.numColumns(); col++) {
                rowValues[col] = Double.toString(m.get(row, col));
            }
            rows.add(rowValues);
        }
        cw.writeAll(rows);
        cw.close();
        bw.close();
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

    public static Matrix expandNominalAttributes(Matrix mCompressed, int[] numCategories) {
        if (numCategories == null) {
            numCategories = new int[mCompressed.numColumns()];
            Arrays.fill(numCategories, 0);
        }

        int numExamples = mCompressed.numRows();
        int numColumnsExpanded = 0;
        for (int numCat:numCategories) {
            numColumnsExpanded += numCat > 0 ? numCat : 1;
        }

        Matrix mExpanded = new Matrix(numExamples, numColumnsExpanded);
        int expandedCol = 0;
        for (int compressedCol = 0; compressedCol < mCompressed.numColumns(); compressedCol++) {
            if (numCategories[compressedCol] <= 1) {
                // Numeric attribute.
                // Copy values
                for (int row = 0; row < numExamples; row++) {
                    mExpanded.set(row, expandedCol, mCompressed.get(row, compressedCol));
                }
                expandedCol++;
            } else {
                // Nominal attribute.
                // Expand values to groups of booleans
                for (int cat = 0; cat < numCategories[compressedCol]; cat++) {
                    for (int row = 0; row < numExamples; row++) {
                        double value = cat == mCompressed.get(row, compressedCol) ? 1.0 : 0.0;
                        mExpanded.set(row, expandedCol, value);
                    }
                    expandedCol++;
                }
            }
        }
        return mExpanded;
    }

    public static Matrix compressNominalAttributes(Matrix mExpanded, int[] numCategories) {
        if (numCategories == null) {
            numCategories = new int[mExpanded.numColumns()];
            Arrays.fill(numCategories, 0);
        }

        int numExamples = mExpanded.numRows();
        int numColumnsCompressed = numCategories.length;


        // Get all nominal values and compress them from groups of booleans to numeric values.
        Matrix mCompressed = new Matrix(numExamples, numColumnsCompressed);
        int expandedCol = 0;
        for (int compressedCol = 0; compressedCol < mCompressed.numColumns(); compressedCol++) {
            if (numCategories[compressedCol] < 1) {
                // Numeric attribute.
                // Copy values
                for (int row = 0; row < numExamples; row++) {
                    mCompressed.set(row, compressedCol, mExpanded.get(row, expandedCol));
                }
                expandedCol++;
            } else {
                // Nominal attribute.
                // Compress values from groups of booleans to single numeric value.
                for (int row = 0; row < numExamples; row++) {
                    mCompressed.set(row, compressedCol, -1);
                    for (int cat = 0; cat < numCategories[compressedCol]; cat++) {
                        // Find the category with 1.0 boolean
                        if (mExpanded.get(row, expandedCol) == 1.0) {
                            mCompressed.set(row, compressedCol, cat);
                        }
                    }
                    expandedCol++;
                }
            }
        }
        return mCompressed;
    }



	/**
     *
	 */

    /**
     * Split a dataset into traingin, crossvalidation and testing data.
     *
 	 * NOTE: If you normalize the input, normalize ONLY on the training data, NOT on the whole data set!
     *
     * @param m
     * @param crossValidationPercent
     * @param testPercent
     * @return Matrix
     */
	public static Matrix[] split(Matrix m, float crossValidationPercent, float testPercent)
	{
		ArrayList<Integer> rowIndexes = new ArrayList<>();
		for (int ri = 0; ri < m.numRows(); ri++) {
			rowIndexes.add(ri);
		}
		Collections.shuffle(rowIndexes);

		int numCVRows = Math.round(m.numRows()*crossValidationPercent/100);
		int numTestRows = Math.round(m.numRows()*testPercent/100);
		int numTrainRows = m.numRows() - numCVRows - numTestRows;

		Matrix trainMatrix = new Matrix(numTrainRows, m.numColumns());
		Matrix cvMatrix = new Matrix(numCVRows, m.numColumns());
		Matrix testMatrix = new Matrix(numTestRows, m.numColumns());

		Iterator<Integer> mRowsIter = rowIndexes.iterator();

		for (int row = 0; row < trainMatrix.numRows(); row++) {
			int mRow = mRowsIter.next().intValue();
			for (int col = 0; col < trainMatrix.numColumns(); col++)
			{
				double value = m.get(mRow, col);
				trainMatrix.set(row, col, value);
			}
		}

		for (int row = 0; row < cvMatrix.numRows(); row++) {
			int mRow = mRowsIter.next().intValue();
			for (int col = 0; col < cvMatrix.numColumns(); col++)
			{
				double value = m.get(mRow, col);
				cvMatrix.set(row, col, value);
			}
		}

		for (int row = 0; row < testMatrix.numRows(); row++) {
			int mRow = mRowsIter.next().intValue();
			for (int col = 0; col < testMatrix.numColumns(); col++)
			{
				double value = m.get(mRow, col);
				testMatrix.set(row, col, value);
			}
		}

		return new Matrix[] {trainMatrix, cvMatrix, testMatrix};
	}

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

    public static Matrix[] softMaxJacobians(Matrix m) {
        Matrix h = softmax(m.copy());
        // One output matrix (Jacobian) for each example (input row).
        Matrix[] results = new Matrix[m.numRows()];
        for (int example = 0; example < m.numRows(); example++) {
            Matrix res = new Matrix(m.numColumns(), m.numColumns());

            // Jacobian has:
            // One row for each (softmax) value whos gradient is taken.
            // One columns for each (softmax) value whos change the gradient is taken in respect to.
            for (MatrixElement me:res) {
                int row = me.row();
                int col = me.col();
                double delta = col == row ? 1.0 : 0.0;
                me.set(h.get(example, row)*(delta-h.get(example, col)));
            }

            results[example] = res;
        }
        return results;
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

	public static double getAverage(Matrix m, int col) {
		double sum = 0.0;
        for (int row = 0; row < m.numRows() ; row++) {
            sum += m.get(row, col);
        }
		return sum / m.numRows();
	}
	

	public static double getStandardDeviation(Matrix m, int col) {
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

		return largestValue - smallestValue;
	}
	
	public static double normalizeData(double x, double average, double standardDeviation) {
        if (standardDeviation == 0.0) {
            standardDeviation = 1.0;
        }
		return (x-average)/standardDeviation;
	}
	
	public static void normalizeData(Matrix x, int col, double average, double standardDeviation) {
        // Avoid division by zero if standard deviation is zero.
        if (standardDeviation == 0.0) {
            standardDeviation = 1.0;
        }
        for (int row = 0; row < x.numRows(); row++)	{
            x.set(row, col, (x.get(row, col)-average)/standardDeviation);
        }
	}

}
