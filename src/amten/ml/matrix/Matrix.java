package amten.ml.matrix;

import no.uib.cipr.matrix.DenseMatrix;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Wrapper class around MTJs Matrix.
 * @link https://github.com/fommil/matrix-toolkits-java
 *
 * Having a wrapper class makes it easier to change underlying matrix lib.
 * Also, implements a few methods lacking in MTJ
 *
 * @author Johannes Amtén
 *
 */
public class Matrix implements Iterable<MatrixElement>, Serializable {

    private DenseMatrix myMatrix;


    public Matrix (int numRows, int numColumns) {
        myMatrix = new DenseMatrix(numRows, numColumns);
    }

    public Matrix (double[][] values) {
        myMatrix = new DenseMatrix(values);
    }

    private Matrix (DenseMatrix m) {
        myMatrix = m;
    }

    public int numRows() {
        return myMatrix.numRows();
    }

    public int numColumns() {
        return myMatrix.numColumns();
    }

    public Matrix copy() {
        DenseMatrix m = new DenseMatrix(numRows(), numColumns());
        System.arraycopy(myMatrix.getData(), 0, m.getData(), 0, m.getData().length );
        return new Matrix(m);
    }

    public double get(int row, int col) {
        return myMatrix.get(row, col);
    }

    public double[] getRow(int row) {
        double[] data = new double[numColumns()];
        for (int col = 0; col < numColumns(); col++) {
            data[col] = get(row, col);
        }
        return data;
    }

    public double[] getCol(int col) {
        double[] data = new double[numRows()];
        for (int row = 0; row < numRows(); row++) {
            data[row] = get(row, col);
        }
        return data;
    }

    public void set(int row, int col, double v) {
        myMatrix.set(row, col, v);
    }

    public void fill(double value) {
        Arrays.fill(myMatrix.getData(), value);
    }

    private void checkSize(Matrix m2) {
        if (numRows() != m2.numRows())
            throw new IndexOutOfBoundsException("A.numRows != B.numRows ("
                    + numRows() + " != " + m2.numRows() + ")");
        if (numColumns() != m2.numColumns())
            throw new IndexOutOfBoundsException(
                    "A.numColumns != B.numColumns (" + numColumns() + " != "
                            + m2.numColumns() + ")");
    }

    public Matrix add(double c, Matrix m2) {
        checkSize(m2);
        double[] d = getData();
        double[] d2 = m2.getData();
        for (int i = 0; i < d.length; i++) {
            d[i] += c*d2[i];
        }
        return this;
    }

    public Matrix add(double c) {
        double[] d = getData();
        for (int i = 0; i < d.length; i++) {
            d[i] += c;
        }
        return this;
    }

    public void add(int row, int col, double v) {
        set(row, col, get(row, col) + v);
    }

    public Matrix mult(Matrix m2) {
        return mult(m2, new Matrix(numRows(), m2.numColumns()));
    }

    public Matrix mult(Matrix m2, Matrix res) {
        myMatrix.mult(m2.myMatrix, res.myMatrix);
        return res;
    }

    public Matrix multElements(Matrix m2) {
        return multElements(m2, new Matrix(numRows(), numColumns()));
    }

    public Matrix multElements(Matrix m2, Matrix res) {
        checkSize(m2);
        checkSize(res);
        double[] d = getData();
        double[] d2 = m2.getData();
        double[] dres = res.getData();
        for (int i = 0; i < d.length; i++) {
            dres[i] = d[i]*d2[i];
        }
        return res;
    }

    public Matrix trans1mult(Matrix m2) {
        return trans1mult(m2, new Matrix(numColumns(), m2.numColumns()));
    }

    public Matrix trans1mult(Matrix m2, Matrix res) {
        myMatrix.transAmult(m2.myMatrix, res.myMatrix);
        return res;
    }

    public Matrix trans2mult(Matrix m2) {
        return trans2mult(m2, new Matrix(numRows(), m2.numRows()));
    }

    public Matrix trans2mult(Matrix m2, Matrix res) {
        myMatrix.transBmult(m2.myMatrix, res.myMatrix);
        return res;
    }

    public Matrix scale(double c) {
        double[] d = getData();
        for (int i = 0; i < d.length; i++) {
            d[i] = d[i]*c;
        }
        return this;
    }

    public double[] getData() {
        return myMatrix.getData();
    }

    public Matrix addColumns(Matrix m2) {
        if (numRows() != m2.numRows())
            throw new IndexOutOfBoundsException("A.numRows != B.numRows ("
                    + numRows() + " != " + m2.numRows() + ")");
        double[] d1 = getData();
        double[] d2 = m2.getData();
        Matrix m3 = new Matrix(numRows(), numColumns() + m2.numColumns());
        double[] d3 = m3.getData();
        System.arraycopy(d1, 0, d3, 0, d1.length);
        System.arraycopy(d2, 0, d3, d1.length, d2.length);

        return m3;
    }

    public Matrix getColumns(int startCol, int endCol) {
        endCol = endCol==-1 ? numColumns()-1 : endCol;
        Matrix m2 = new Matrix(numRows(), endCol-startCol+1);
        System.arraycopy(myMatrix.getData(), startCol*numRows(), m2.getData(), 0, m2.getData().length);
        return m2;
    }

    public Matrix getRows(int startRow, int endRow) {
        endRow = endRow==-1 ? numRows()-1 : endRow;
        Matrix m2 = new Matrix(endRow-startRow+1, numColumns());
        for (int row = 0; row < m2.numRows(); row++) {
            for (int col = 0; col < m2.numColumns(); col++) {
                m2.set(row, col, get(row+startRow, col));
            }
        }
        return m2;
    }

    public Iterator<MatrixElement> iterator() {
        return new Iterator<MatrixElement>() {
            private MatrixElement me = new MatrixElement(Matrix.this);

            public boolean hasNext() {
                return me.myPos < me.myData.length-1;
            }

            public MatrixElement next() {
                me.myPos++;
                return me;
            }

            public void remove() {
                throw new UnsupportedOperationException("Nope.");
            }
        };
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numColumns(); col++) {
                sb.append(get(row, col)).append("  ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Needed because MTJs Matrix is not serializable
     */
    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.writeInt(numRows());
        out.writeInt(numColumns());
        out.writeObject(getData());
    }

    /**
     * Needed because MTJs Matrix is not serializable
     */
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        myMatrix = new DenseMatrix(in.readInt (), in.readInt());
        double[] data = (double[]) in.readObject();
        System.arraycopy(data, 0, myMatrix.getData(), 0, data.length);
    }
}
