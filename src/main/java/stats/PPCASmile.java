package stats;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class PPCASmile implements PCA{
    private static final Logger log = LoggerFactory.getLogger(PPCASmile.class);

    private RealMatrix dataMatrix; // A
    private Matrix transformation; // V
    private int N;
    private int M;
    private smile.projection.PPCA pca;

    public PPCASmile(RealMatrix rawDataMatrix) {
        this.dataMatrix = rawDataMatrix;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();

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

    public RealMatrix transform(RealMatrix inputData, int K){
        if (K > Math.min(this.N,this.M)){
            log.warn("Watch your K...K {} M {} N {}", K, this.M, this.N);
        }
        K = Math.min(Math.min(K, this.N), this.M);
        this.pca = new smile.projection.PPCA(dataMatrix.getData(),K);
        this.transformation = new DenseMatrix(pca.getProjection().array());
        return new Array2DRowRealMatrix(pca.project(inputData.getData()));
    }

    //benchmarking PCA class, should never be used.
    public RealMatrix transform(RealMatrix inputData, int K, TransformHistory history) {
        return null;
    }
}
