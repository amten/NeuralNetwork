package amten.ml.matrix;

/**
 * Class used when iterating over elements in a Matrix.
 *
 * @author Johannes Amtén
 *
 */

public class MatrixElement {
    protected double[] myData;
    protected int myPos = -1;
    private int myNumRows;

    public MatrixElement(Matrix m) {
        myData = m.getData();
        myNumRows = m.numRows();
    }

    public int row() {
        return myPos % myNumRows;
    }

    public int col() {
        return myPos / myNumRows;
    }

    public double value() {
        return myData[myPos];
    }

    public void set(double v) {
        myData[myPos] = v;
    }

}
