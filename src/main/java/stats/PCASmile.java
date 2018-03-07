package stats;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import utils.util;

import static utils.util.ASubbMultC;
import static utils.util.centerDataMatrix;
import static utils.util.makeIntList;


public class PCASmile implements PCA{
    private static final Logger log = LoggerFactory.getLogger(PCASmile.class);

    private RealMatrix dataMatrix; // A
    private Matrix transformation; // V
    private int N;
    private int M;
    private smile.projection.PCA pca;
    private RealVector columnMeans;


    public PCASmile(RealMatrix rawDataMatrix) {
        this.dataMatrix = rawDataMatrix;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();

        util.matVec dataMeans = centerDataMatrix(rawDataMatrix);
        this.columnMeans = dataMeans.vector;

        this.pca = new smile.projection.PCA(rawDataMatrix.getData());
        this.transformation = new DenseMatrix(pca.getProjection().array());
    }

    public int getN(){
        return this.N;
    }

    public int getM(){ return this.M; }

    public Matrix getTransformation(){ return this.transformation; }

    public RealMatrix getTransformationMatrix(int K){
        RealMatrix transform = new Array2DRowRealMatrix(Matrices.getArray(transformation));
        return transform.getSubMatrix(0,N-1,0,K-1);
    }

    public RealMatrix transformOld(RealMatrix inputData, int K){
        if (K > Math.min(this.N,this.M)){
            log.warn("Watch your K...K {} M {} N {}", K, this.M, this.N);
        }
        K = Math.min(Math.min(K, this.N), this.M);
        pca.setProjection(K);

        return new Array2DRowRealMatrix(pca.project(inputData.getData()));
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
