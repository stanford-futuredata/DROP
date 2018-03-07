package stats;

import no.uib.cipr.matrix.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static utils.util.*;

import java.util.Random;

public class PCATropp implements PCA{
    private static final Logger log = LoggerFactory.getLogger(PCATropp.class);

    private RealMatrix dataMatrix;
    private RealMatrix centeredDataMatrix;
    private RealVector columnMeans;
    private Matrix transformation;
    private Matrix UMatrix; //needed for work reuse
    private int M;
    private int N;
    private boolean init; //flag to see if this has not been used to transform already
    private int p; //small additional number of columns to pull. Either 5, 10 or L
    private int q; //small number of power iterations. Typically 1-3
    private int highestK; //largest K obtained with this data
    private int highestSmooshK; //largest K obtained with work reuse

    public PCATropp(RealMatrix rawDataMatrix){
        this.p = 5;
        this.q = 1; //checking if this gives good enough results with 1
        this.highestK = 0;
        this.highestSmooshK = 0;
        this.dataMatrix = rawDataMatrix;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();
        this.init = true;

        matVec dataMeans = centerDataMatrix(rawDataMatrix);

        this.centeredDataMatrix = dataMeans.matrix;
        this.columnMeans = dataMeans.vector;
    }

    public int getN(){
        return this.N;
    }

    public int getM(){ return this.M; }

    public Matrix getTransformation(){ return this.transformation; }

    public Matrix getUMatrix() {
        return this.UMatrix;
    }

    public void computeTransformation(int K) {
        if (K > Math.min(this.N, this.M)) {
            //System.out.println(String.format("Tropp Watch your K...K %d M %d Nproc %d", K, this.M, this.N));
        }
        K = Math.min(Math.min(K, this.N), this.M);
        int Kp = Math.min(Math.min(K + p, this.N), this.M);
        DenseMatrix tm; //transformation matrix

        //if the K you want is higher than what you've seen compute transform from scratch
        if (K > highestK){
            highestK = K;
            transformation = new DenseMatrix(N, K);//new Array2DRowRealMatrix(this.N, K);

            DenseMatrix ci; //centeredInput
            DenseMatrix omega = new DenseMatrix(N, Kp); //random initializer matrix
            DenseMatrix Y1 = new DenseMatrix(M, Kp); //intermediate matrix
            DenseMatrix Y2 = new DenseMatrix(N, Kp); //intermediate matrix
            DenseMatrix B = new DenseMatrix(Kp, N); //intermediate matrix


            Random rand = new Random();

            QR qr = new QR(M, Kp);
            SVD svd = new SVD(Kp, N);

            //generate gaussian random initialization matrix
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < Kp; j++)
                    omega.set(i, j, rand.nextGaussian());
            }

            ci = new DenseMatrix(centeredDataMatrix.getData());

            //run some modded power iteration
            ci.mult(omega, Y1); // mxn times nx(k+p), A*Omega
            for (int i = 0; i < q; i++) {
                ci.transAmult(Y1, Y2); // (mxn)' times mx(k+p)  A' * A*Omega
                ci.mult(Y2, Y1);       //  mxn times nx(k+p)  A * A'AOmega

                qr.factor(Y1);          // QR(AA'AOmega)
                Y1 = qr.getQ(); //Q in the algo
            }

            //SVD to get the top K
            Y1.transAmult(ci, B);
            //svd = svd.factor(B);
            try {
                svd = svd.factor(B);
                tm = svd.getVt();
                tm.transpose();
                transformation = Matrices.getSubMatrix(tm, makeIntList(N), makeIntList(K)).copy();//new Array2DRowRealMatrix(Matrices.getArray(tm)).getSubMatrix(0, N-1, 0, K-1);
            } catch (NotConvergedException ie) {
                ie.printStackTrace();
            }
        }
    }

    public void computeTransformationForReuse(int K){
        if (K > Math.min(this.N, this.M)) {
            //System.out.println(String.format("Tropp Reuse Watch your K...K %d M %d Nproc %d", K, this.M, this.N));
        }
        K = Math.min(Math.min(K, this.N), this.M);
        int Kp = Math.min(Math.min(K + p, this.N), this.M);
        DenseMatrix tm; //transformation matrix

        //if the K you want is higher than what you've seen compute transform from scratch
        if (K > highestK){
            highestK = K;
            transformation = new DenseMatrix(N, K);//new Array2DRowRealMatrix(this.N, K);

            DenseMatrix ci; //centeredInput
            DenseMatrix omega = new DenseMatrix(N, Kp); //random initializer matrix
            DenseMatrix Y1 = new DenseMatrix(M, Kp); //intermediate matrix
            DenseMatrix Y2 = new DenseMatrix(N, Kp); //intermediate matrix
            DenseMatrix B = new DenseMatrix(Kp, N); //intermediate matrix

            UMatrix = new DenseMatrix(M, Kp);


            Random rand = new Random();

            QR qr = new QR(M, Kp);
            SVD svd = new SVD(Kp, N);

            //generate gaussian random initialization matrix
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < Kp; j++)
                    omega.set(i, j, rand.nextGaussian());
            }

            ci = new DenseMatrix(centeredDataMatrix.getData());

            //run some modded power iteration
            ci.mult(omega, Y1); // mxn times nx(k+p)
            for (int i = 0; i < q; i++) {
                ci.transAmult(Y1, Y2); // (mxn)' times mx(k+p)
                ci.mult(Y2, Y1);       //  mxn times nx(k+p)

                qr.factor(Y1);
                Y1 = qr.getQ();
            }

            //SVD to get the top K
            Y1.transAmult(ci, B);
            //svd = svd.factor(B);
            try {
                svd = svd.factor(B);
                tm = svd.getVt();
                tm.transpose();
                Y1.mult(svd.getU(), UMatrix);
                UMatrix = Matrices.getSubMatrix(UMatrix, makeIntList(M), makeIntList(K)).copy();
                transformation = Matrices.getSubMatrix(tm, makeIntList(N), makeIntList(K)).copy();//new Array2DRowRealMatrix(Matrices.getArray(tm)).getSubMatrix(0, N-1, 0, K-1);
            } catch (NotConvergedException ie) {
                ie.printStackTrace();
            }
        }

    }

    public void computeTransformation(int K, TransformHistory history) {
        if (K > Math.min(Math.max(history.getSize()+this.N,this.N), this.M)) {
            //System.out.println(String.format("Tropp History Watch your K...K %d M %d Nproc %d", K, this.M, this.N));
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
        Matrix tm; //transformation matrix

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
