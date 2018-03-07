package stats;

import com.google.common.base.Stopwatch;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.QR;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static utils.util.*;

/**
 * intended to be run once only, on a known k for meaningful progress checking
 */
public class PCAOjas implements PCA {
    private static final Logger log = LoggerFactory.getLogger(PCAOjas.class);

    private RealMatrix dataMatrix;
    private RealMatrix centeredDataMatrix;
    private RealVector columnMeans;
    private Matrix transformation;
    private int M;
    private int N;

    private int highestK; //largest K obtained with this data
    private int highestSmooshK; //largest K obtained including work reuse

    private int batchSize;
    private int iters;
    private double learningRate;
    private int freq; //how often to record stats (every freq iterations)

    private Map<Integer,Long> iterationRuntime; //map with iteration:runtime
    private Map<Long,RealMatrix> progressRuntime; //map with runtime:corresponding transformation matrix after that iter of SGD
    private Map<Long,Double> progressLBR; //map with runtime:corresponding transformation matrix after that iter of SGD
    private Map<Long, Double> progressConvergence;

    public PCAOjas(RealMatrix rawDataMatrix){
        this(rawDataMatrix, 1, 100000, 1e-5, 1000);
    }

    public PCAOjas(RealMatrix rawDataMatrix, int batchSize, int iters, double learningRate, int freq) {
        this.progressLBR = new HashMap<>();
        this.progressRuntime = new HashMap<>();
        this.iterationRuntime = new HashMap<>();
        this.progressConvergence = new HashMap<>();

        this.dataMatrix = rawDataMatrix;
        this.batchSize = batchSize;
        this.iters = iters;
        this.learningRate = learningRate;
        this.freq = freq;
        this.M = rawDataMatrix.getRowDimension();
        this.N = rawDataMatrix.getColumnDimension();
        this.highestK = 0;
        this.highestSmooshK = 0;
        matVec dataMeans = centerDataMatrix(rawDataMatrix);

        this.centeredDataMatrix = dataMeans.matrix;
        this.columnMeans = dataMeans.vector;
    }

    //right now just computes on 300 pairs. This really shouldn't be in here but oh well...
    public double computeLBR(RealMatrix originalData, RealMatrix transformedData) {
        double threshold = 1.96;
        int numPairs = 300;
        int K = transformedData.getColumnDimension();
        M = originalData.getRowDimension();
        N = originalData.getColumnDimension();

        int[][] indicesAB = randPairs(numPairs, M);
        int[] indicesA = indicesAB[0];
        int[] indicesB = indicesAB[1];
        int[] kIndices = makeIntList(K);
        int[] allIndicesN = makeIntList(N);

        RealVector transformedDists;
        RealVector trueDists;

        List<Double> LBRs;
        double mean = 0;
        double std = 0;
        double slop;

        transformedDists = calcDistances(transformedData.getSubMatrix(indicesA, kIndices), transformedData.getSubMatrix(indicesB, kIndices));
        trueDists = calcDistances(originalData.getSubMatrix(indicesA, allIndicesN), originalData.getSubMatrix(indicesB, allIndicesN));
        LBRs = calcLBRList(trueDists, transformedDists);
        return confInterval(LBRs, threshold)[1];
    }


    public int getN(){
        return this.N;
    }

    public int getM(){ return this.M; }

    public Matrix getTransformation(){ return this.transformation; }

    public Map getIterationProgress(){
        return this.iterationRuntime;}

    public Map getTransformationProgress(){
        return progressRuntime;
    }

    public Map getLBRProgress() {
        return progressLBR;
    }

    public Map getConvergenceProgress() {
        return progressConvergence;
    }

    private void warmup(){
        int K = 3;
        int[] allN = makeIntList(N);
        int[] batch;
        DenseMatrix ci; //centeredInput in batches
        DenseMatrix x = new DenseMatrix(N, K); //random initializer matrix
        DenseMatrix temp = new DenseMatrix(batchSize,K);

        Random rand = new Random();
        QR qr = new QR(N,K);

        //generate gaussian random initialization matrix
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++)
                x.set(i, j, rand.nextGaussian());
        }

        //orthogonalize
        qr.factor(x);
        x = qr.getQ();

        //run SGD
        for (int i = 0; i < 5; i++) {
            batch = randIntsWithoutReplacement(M,batchSize);
            ci = new DenseMatrix(centeredDataMatrix.getSubMatrix(batch,allN).getData());
            ci.mult(x, temp); //temp = X*x
            ci.transAmultAdd(learningRate, temp, x); // x = x + eta(X.T*X*x) = x + eta*(X.T*temp)
            qr.factor(x);
            x = qr.getQ();
        }
    }

    //computes and stores LBR progress
    public void updateTransformProgress(RealMatrix inputData, int K, Boolean svd) {
        DenseMatrix ci; // centered input
        DenseMatrix out = new DenseMatrix(inputData.getRowDimension(),K);
        DenseMatrix x; //transformation
        RealVector currVec;
        RealMatrix centeredInput = new Array2DRowRealMatrix(inputData.getData());

        RealMatrix trueTransform = new Array2DRowRealMatrix();
        if (svd){
            //compute true PCA via SVD
            PCASVD truePCA = new PCASVD(this.dataMatrix);
            trueTransform = truePCA.getTransformationMatrix(K);
        }

        //center, transform input data
        for (int i = 0; i < this.N; i++) {
            currVec = inputData.getColumnVector(i);
            currVec.mapSubtractToSelf(this.columnMeans.getEntry(i));
            centeredInput.setColumn(i, currVec.toArray());
        }
        ci = new DenseMatrix(centeredInput.getData());

        for (Map.Entry<Long, RealMatrix> entry : progressRuntime.entrySet()) {
            x = new DenseMatrix(entry.getValue().getData());
            ci.mult(x, out);
            progressLBR.put(entry.getKey(), computeLBR(inputData, new Array2DRowRealMatrix(Matrices.getArray(out))));
            if (svd) {
                progressConvergence.put(entry.getKey(), computeSine(trueTransform,entry.getValue()));
            }
        }
    }

    private void computeTransformation(int K){
        /* In python, for reference:
        x = np.random.rand(X.shape[1],kkk)
        x,_ = np.linalg.qr(x)
        for ii in range(1,epoch):
            ids = np.random.randint(0,X.shape[0],b)
            x += eta*np.dot(X[ids,:].T,np.dot(X[ids,:],x))
            x,_ = np.linalg.qr(x)
        */
        if (K > Math.min(this.N, this.M)) {
            log.warn("Watch your K...K {} M {} Nproc {}", K, this.M, this.N);
        }
        K = Math.min(Math.min(K, this.N), this.M);

        if (K > highestK) {
            Stopwatch sw = Stopwatch.createUnstarted();
            long currTime;
            int[] allN = makeIntList(N);
            int[] batch;
            DenseMatrix ci; //centeredInput in batches
            DenseMatrix x = new DenseMatrix(N, K); //random initializer matrix
            DenseMatrix temp = new DenseMatrix(batchSize, K);

            Random rand = new Random();
            QR qr = new QR(N, K);

            //todo: decide if you need to remove this or not
            //warmup();

            sw.start();
            //generate gaussian random initialization matrix
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < K; j++)
                    x.set(i, j, rand.nextGaussian());
            }

            //orthogonalize
            qr.factor(x);
            x = qr.getQ();

            //run SGD
            for (int i = 0; i < iters; i++) {
                sw.stop();
                if (i % freq == 0) {
                    currTime = sw.elapsed(TimeUnit.MILLISECONDS);
                    progressRuntime.put(currTime, new Array2DRowRealMatrix(Matrices.getArray(x)));
                    iterationRuntime.put(i, currTime);
                }
                batch = randIntsWithoutReplacement(M, batchSize);
                sw.start(); //not including overhead from sampling out the points....
                ci = new DenseMatrix(centeredDataMatrix.getSubMatrix(batch, allN).getData());
                ci.mult(x, temp); //temp = X*x
                ci.transAmultAdd(learningRate, temp, x); // x = x + eta(X.T*X*x) = x + eta*(X.T*temp)
                qr.factor(x);
                x = qr.getQ();
            }
            this.transformation = x; //Matrices.getSubMatrix(x, makeIntList(N), makeIntList(K));
            //return x;
        }
    }

    public void computeTransformation(int K, TransformHistory history) {
        if (K > Math.min(Math.max(history.getSize()+this.N,this.N), this.M)) {
            log.warn("Watch your K...K {} M {} Nproc {}", K, this.M, this.N);
        }
        int thisK = Math.min(Math.min(K, this.N), this.M);
        //K = Math.min(Math.min(K, this.N), this.M);
        if (K > highestSmooshK){
            highestSmooshK = K;
            computeTransformation(thisK);
            transformation = history.smoosh(Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(thisK)).copy(),K); //TODO: find and kill submatrix stuff after works
        }
    }

    //right now, no sampling done on this guy's input, and it's only used for benchmarking purposes
    public RealMatrix transform(RealMatrix inputData, int K) {
        Matrix tm;

        computeTransformation(K); // updating transformation
        tm = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(K)).copy();
        return ASubbMultC(inputData, this.columnMeans, tm);
    }

    public RealMatrix transform(RealMatrix inputData, int K, TransformHistory history) {
        Matrix tm; //transformation matrix

        computeTransformation(K, history); //update transformation with history
        tm = Matrices.getSubMatrix(transformation, makeIntList(N), makeIntList(K)).copy(); //retrieve transform
        return ASubbMultC(inputData, this.columnMeans, tm); // apply transform
    }

}
