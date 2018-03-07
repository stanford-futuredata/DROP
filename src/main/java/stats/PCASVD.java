package stats;

import no.uib.cipr.matrix.*;
import org.apache.commons.math3.linear.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static utils.util.*;



public class PCASVD implements PCA{
    private static final Logger log = LoggerFactory.getLogger(PCASVD.class);

    private RealMatrix dataMatrix; // A
    private RealMatrix centeredDataMatrix; // X
    private Matrix transformation; // V
    private  Matrix UMatrix;
    private RealVector columnMeans;
    private SVD svd; //gives X = UDV', U=mxp D=pxp V' = pxn
    private double[] spectrum;
    private int N;
    private int M;
    private int P;

    public PCASVD(RealMatrix rawDataMatrix) {
        this.dataMatrix = rawDataMatrix;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();

        matVec dataMeans = centerDataMatrix(rawDataMatrix);

        this.centeredDataMatrix = dataMeans.matrix;
        this.columnMeans = dataMeans.vector;

        svd = new SVD(M, N);
        try {
            DenseMatrix cdm = new DenseMatrix(centeredDataMatrix.getData());
            svd = svd.factor(cdm);
            DenseMatrix tm = svd.getVt();
            spectrum = svd.getS();
            tm.transpose();
            transformation = tm;
            UMatrix = svd.getU();
            P = transformation.numColumns(); //TODO: remember this used to be rowdim
        } catch (NotConvergedException ie) {
            ie.printStackTrace();
        }
    }

    public int getN(){
        return this.N;
    }

    public int getM(){ return this.M; }

    public double[] getSpectrum() { return this.spectrum; }

    public Matrix getTransformation(){ return this.transformation; }

    public Matrix getUMatrix(){ return this.UMatrix; }

    public RealMatrix getTransformationMatrix(int K){
        RealMatrix transform = new Array2DRowRealMatrix(Matrices.getArray(transformation));
        return transform.getSubMatrix(0,N-1,0,K-1);
    }

    public RealMatrix transform(RealMatrix inputData, int K){
        if (K > Math.min(this.N,this.M)){
            log.warn("Watch your K...K {} M {} N {}", K, this.M, this.N);
        }
        K = Math.min(Math.min(K, this.N), this.M);
        Matrix t = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(K)).copy();
        return ASubbMultC(inputData, this.columnMeans, t); // apply transform
    }

    public RealMatrix transform(RealMatrix inputData, int K, TransformHistory history) {
        if (K > Math.min(Math.max(history.getSize()+this.N,this.N), this.M)) {
            log.warn("Watch your K...K {} M {} Nproc {}", K, this.M, this.N);
        }
        int thisK = Math.min(Math.min(K, this.N), this.M);
        Matrix t = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(thisK)).copy();
        t = history.smoosh(Matrices.getSubMatrix(t, makeIntList(N), makeIntList(thisK)).copy(),K); //TODO: find and kill submatrix stuff after works
        t = Matrices.getSubMatrix(t, makeIntList(N), makeIntList(K)).copy();
        return ASubbMultC(inputData, this.columnMeans, t); // apply transform
    }
}

