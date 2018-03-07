package utils;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import java.util.*;

/**
 * Created by sahaana on 12/30/17.
 */
public final class util {

    private util(){
    }

    // Class to return both a matrix and vector
    public final static class matVec{
        public RealMatrix matrix;
        public RealVector vector;

        matVec(RealMatrix mat, RealVector vec)
        {
            this.matrix = mat;
            this.vector = vec;
        }
    }

    // Returns a centered data matrix and column means given an input data matrix (i.e., [ones] * return = 0)
    public static matVec centerDataMatrix(RealMatrix inputData) {
        int M = inputData.getRowDimension();
        int N = inputData.getColumnDimension();
        double mean;
        RealVector currVec;

        RealMatrix centeredDataMatrix = new Array2DRowRealMatrix(M, N);
        RealVector columnMeans = new ArrayRealVector(N);


        //TODO: put in util
        for (int i = 0; i < N; i++){
            currVec = inputData.getColumnVector(i);
            mean = 0;
            for (double entry: currVec.toArray()){
                mean += entry;
            }
            mean /= M;
            columnMeans.setEntry(i, mean);
            currVec.mapSubtractToSelf(mean);
            centeredDataMatrix.setColumnVector(i, currVec);
        }

        return new matVec(centeredDataMatrix, columnMeans);
    }

    //computes ("A-b")*C where A-b is "column-wise" subtraction
    public static RealMatrix ASubbMultC(RealMatrix A, RealVector b, Matrix C) {
        RealMatrix centeredOutput = new Array2DRowRealMatrix(A.getData());
        RealVector currVec;
        Matrix co; //centeredOutput
        Matrix transformedData = new DenseMatrix(A.getRowDimension(), C.numColumns());

        //center, transform input data and return
        for (int i = 0; i < C.numRows(); i++) {
            currVec = A.getColumnVector(i);
            currVec.mapSubtractToSelf(b.getEntry(i));
            centeredOutput.setColumn(i, currVec.toArray());
        }
        co = new DenseMatrix(centeredOutput.getData());

        co.mult(C, transformedData);

        return new Array2DRowRealMatrix(Matrices.getArray(transformedData));
    }

    // Computes the trace(1-(W.T*U).^2)/num_cols. i.e., "average squared sine" between spaces
    public static double computeSine(RealMatrix U, RealMatrix W){
        RealMatrix temp = W.transpose().multiply(U);
        for (int i = 0; i<U.getColumnDimension(); i++) {
            temp.multiplyEntry(i, i, temp.getEntry(i, i));
        }
        return temp.scalarMultiply(-1.0).scalarAdd(1.0).getTrace()/U.getColumnDimension();
    }

    // Returns an arraylist with integers from 0 to n-1
    public static ArrayList makeArrayList(int n){
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(i, i);
        }
        return list;
    }

    // Returns a list with integers from 0 to n-1
    public static int[] makeIntList(int n) {
        int[] list = new int[n];
        for (int i = 0; i < n; i++) {
            list[i] = i;
        }
        return list;
    }

    // Returns numInts integers drawn randomly without replacement from 1 to n
    public static int[] randIntsWithoutReplacement(int n, int numInts) {
        int[] ids = new int[numInts];
        Random rand = new Random();
        int curr;
        ArrayList<Integer> allIds = makeArrayList(n);

        for(int i = 0; i < numInts; i++) {
            curr = rand.nextInt(allIds.size());
            ids[i] = allIds.get(curr);
            allIds.remove(curr);
        }
        return ids;
    }

    //Generates two lists, (A,B) of ints from 0 to M-1, where A[i] != B[i]
    public static int[][] randPairs(int numPairs, int M){
        int[][] indices = new int[2][numPairs];
        Random rand = new Random();

        for (int i = 0; i < numPairs; i++) {
            indices[0][i] = rand.nextInt(M);
            indices[1][i] = rand.nextInt(M);
            while (indices[0][i] == indices[1][i]) {
                indices[0][i] = rand.nextInt(M);
            }
        }
        return indices;
    }

    public static double[] confInterval(List<Double> values, double q){
        double mean = 0;
        double std = 0;
        double slop;

        for(double l: values){
            mean += l;
        }
        mean /= values.size();

        for(double l: values){
            std += (l - mean)*(l - mean);
        }
        std = Math.sqrt(std/values.size());
        slop = (q*std)/Math.sqrt(values.size());
        return new double[] {mean-slop, mean, mean+slop, std*std};
    }


    public static List<Double> calcLBRList(RealVector trueDists, RealVector transformedDists){
        int num_entries = trueDists.getDimension();
        List<Double> lbr = new ArrayList<>();
        for (int i = 0; i < num_entries; i++) {
            if (transformedDists.getEntry(i) == 0){
                if (trueDists.getEntry(i) == 0) lbr.add(1.0); //they were same to begin w/, so max of 1
                else lbr.add(0.0); //can never be negative, so lowest
            }
            else lbr.add(transformedDists.getEntry(i)/trueDists.getEntry(i));
        }
        return lbr;
    }

    public static RealVector calcDistances(RealMatrix dataA, RealMatrix dataB){
        int rows = dataA.getRowDimension();
        RealMatrix differences = dataA.subtract(dataB);
        RealVector distances = new ArrayRealVector(rows);
        RealVector currVec;
        for (int i = 0; i < rows; i++){
            currVec = differences.getRowVector(i);
            distances.setEntry(i, currVec.getNorm());
        }
        return distances;
    }

    public static int[] complexArgSort(Complex[] in, boolean ascending) {
        Integer[] indices = new Integer[in.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return (ascending ? 1 : -1) * Double.compare(in[o1].abs(), in[o2].abs());
            }
        });
        return toPrimitive(indices);
    }

    //input must be even length array in [re[0], im[0],...,re[k], im[k]]
    public static int[] complexArgSort(double[] in, boolean ascending) {
        Integer[] indices = new Integer[in.length/2];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return (ascending ? 1 : -1) * Double.compare(abs(in[2*o1],in[2*o1+1]), abs(in[2*o2],in[2*o2+1]));
            }
        });
        return toPrimitive(indices);
    }

    public static double abs(double real, double imaginary) {
        double q;
        if(FastMath.abs(real) < FastMath.abs(imaginary)) {
            if(imaginary == 0.0D) {
                return FastMath.abs(real);
            } else {
                q = real / imaginary;
                return FastMath.abs(imaginary) * FastMath.sqrt(1.0D + q * q);
            }
        } else if(real == 0.0D) {
            return FastMath.abs(imaginary);
        } else {
            q = imaginary / real;
            return FastMath.abs(real) * FastMath.sqrt(1.0D + q * q);
        }
    }

    public static int[] argSort(int[] in, boolean ascending) {
        Integer[] indices = new Integer[in.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return (ascending ? 1 : -1) * Integer.compare(in[o1], in[o2]);
            }
        });
        return toPrimitive(indices);
    }

    public static int[] toPrimitive(Integer[] in) {
        int[] out = new int[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i];
        }
        return out;
    }


    public static int[] intListToPrimitive(List<Integer> in) {
        int[] out = new int[in.size()];
        for (int i = 0; i < in.size(); i++) {
            out[i] = in.get(i).intValue();
        }
        return out;
    }

    public static double[] doubleListToPrimitive(List<Double> in) {
        double[] out = new double[in.size()];
        for (int i = 0; i < in.size(); i++) {
            out[i] = in.get(i).doubleValue();
        }
        return out;
    }

    public static int[] argSort(Double[] in, boolean ascending) {
        Integer[] indices = new Integer[in.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return (ascending ? 1 : -1) * Double.compare(in[o1], in[o2]);
            }
        });
        return toPrimitive(indices);
    }

    public static double checkMatrices(RealMatrix A, RealMatrix B) {
        RealMatrix diff = A.subtract(B);
        return diff.getFrobeniusNorm();
    }

    public static double checkMatrices(Matrix A, Matrix B) {
        return checkMatrices(new Array2DRowRealMatrix(Matrices.getArray(A)), new Array2DRowRealMatrix(Matrices.getArray(B)));
    }

    public static double[] stringToDoubleArray(String[] A) {
        double[] output = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            output[i] = Double.parseDouble(A[i]);
        }
        return output;
    }
}
