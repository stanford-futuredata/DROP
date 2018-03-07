package stats;

import no.uib.cipr.matrix.*;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static utils.util.*;

import java.util.Random;

public class PCAFastPowerIteration implements PCA{
    private static final Logger log = LoggerFactory.getLogger(PCAFastPowerIteration.class);

    private RealMatrix dataMatrix;
    private RealMatrix centeredDataMatrix;
    private RealVector columnMeans;
    private Matrix transformation;
    private int M;
    private int N;
    private int highestK; //highest thing transformed so far, without using reuse
    private int highestSmooshK; //highest thing transformed, period

    public PCAFastPowerIteration(RealMatrix rawDataMatrix){
        this.dataMatrix = rawDataMatrix;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();
        this.highestK = 0;
        this.highestSmooshK = 0;

        matVec dataMeans = centerDataMatrix(rawDataMatrix);

        this.centeredDataMatrix = dataMeans.matrix;

        this.columnMeans = dataMeans.vector;
    }

    public int getN(){
        return this.N;
    }

    public int getM(){ return this.M; }

    public Matrix getTransformation(){ return this.transformation; }

    public void computeTransformation(int K) {
        if (K > Math.min(this.N, this.M)) {
            //System.out.println(String.format("Fast Watch your K...K %d M %d Nproc %d", K, this.M, this.N));
        }
        K = Math.min(Math.min(K, this.N), this.M);
        DenseMatrix tm; //transformation matrix

        //if the K you want is higher than what you've seen compute transform from scratch
        if (K > highestK) {
            highestK = K;
            transformation = new DenseMatrix(N, K);//new Array2DRowRealMatrix(this.N, K);

            DenseMatrix ci; //centeredInput
            DenseMatrix omega = new DenseMatrix(N, K); //random initializer matrix
            DenseMatrix Y1 = new DenseMatrix(M, K); //intermediate matrix
            DenseMatrix Y2 = new DenseMatrix(N, K);

            Random rand = new Random();

            QR qr = new QR(N, K);

            //generate gaussian random initialization matrix
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < K; j++)
                    omega.set(i, j, rand.nextGaussian());
            }

            //run one step power iteration
            ci = new DenseMatrix(centeredDataMatrix.getData());
            ci.mult(omega, Y1);
            ci.transAmult(Y1, Y2);
            qr.factor(Y2);
            tm = qr.getQ();
            transformation = Matrices.getSubMatrix(tm, makeIntList(N), makeIntList(K)).copy();//new Array2DRowRealMatrix(Matrices.getArray(tm)).getSubMatrix(0, N-1, 0, K-1);
        }
    }

    public void computeTransformation(int K, TransformHistory history) {
        if (K > Math.min(Math.max(history.getSize()+this.N,this.N), this.M)) {
            //System.out.println(String.format("Fast History Watch your K...K %d M %d Nproc %d", K, this.M, this.N));
        }
        int thisK = Math.min(Math.min(K, this.N), this.M);
        //K = Math.min(Math.min(K, this.N), this.M);
        if (K > highestSmooshK){
            highestSmooshK = K;
            computeTransformation(thisK);
            transformation = history.smoosh(Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(thisK)).copy(),K); //TODO: find and kill submatrix stuff after works
        }
    }

    public RealMatrix transform(RealMatrix inputData, int K) {
        Matrix tm;

        computeTransformation(K); //update transformation
        tm = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(K)).copy(); //retrieve transform
        return ASubbMultC(inputData, this.columnMeans, tm); // apply transform
    }

    public RealMatrix transform(RealMatrix inputData, int K, TransformHistory history) {
        Matrix tm; //transformation matrix

        computeTransformation(K, history); //update transformation with history
        tm = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(K)).copy(); //retrieve transform
        return ASubbMultC(inputData, this.columnMeans, tm); // apply transform
    }
}
