package optimizer;

import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import static utils.util.*;

public abstract class SkiingOptimizer {
    private static final Logger log = LoggerFactory.getLogger(SkiingOptimizer.class);
    protected int N; //original data dimension
    protected int M; //original number of training samples
    protected List<Integer> trainList;

    protected int kDiff;
    protected double avgKDiff;
    protected int[] kDiffList;
    protected double MDDiff;
    protected int prevK;
    protected double prevMDTime;
    protected double[] currKCI;
    protected int numDiffs;

    protected List<Integer> NtList;
    protected Map<Integer, Integer> KList;
    protected Map<Integer, double[]> LBRList;
    protected Map<Integer, Double> trainTimeList;
    protected Map<Integer, Double> predictedTrainTimeList;
    protected Map<Integer, Integer> kPredList;
    protected Map<Integer, Double> trueObjective;
    protected Map<Integer, Double> predictedObjective;
    protected Map<Integer, Double> fObjective;
    protected Map<Integer, Double> dObjective;

    double dropTimeSoFar;


    protected double qThresh;

    protected RealMatrix dataMatrix;

    protected boolean feasible;
    protected boolean firstKDrop;
    protected int lastFeasible;

    protected int Ntdegree;
    //protected int kScaling;
    protected PolynomialCurveFitter fitter;
    protected WeightedObservedPoints MDruntimes;

    //general indices that go 0 to M/N
    protected int[] allIndicesN;
    protected int[] allIndicesM;

    protected boolean opt;
    protected boolean reuse;
    protected boolean sample;
    protected boolean knn;


    public SkiingOptimizer(double qThresh) {
        this.numDiffs = 3; //TODO: 3 to change to general param
        this.qThresh = qThresh;

        this.NtList = new ArrayList<>();
        this.LBRList = new HashMap<>();
        this.KList = new HashMap<>();
        this.trainTimeList = new HashMap<>();
        this.predictedTrainTimeList = new HashMap<>();
        this.trueObjective = new HashMap<>();
        this.predictedObjective = new HashMap<>();
        this.fObjective = new HashMap<>();
        this.dObjective = new HashMap<>();
        this.kPredList = new HashMap<>();

        this.MDDiff = 0;
        this.kDiff = 0;
        this.avgKDiff = Integer.MAX_VALUE;
        this.kDiffList = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE};
        this.prevK = 0;
        this.prevMDTime = 0;
        this.currKCI = new double[]{0, 0, 0};
        this.dropTimeSoFar = 0;

        this.feasible = false;
        this.firstKDrop = true;
        this.lastFeasible = Integer.MAX_VALUE;

        this.Ntdegree = 2;
        this.MDruntimes = new WeightedObservedPoints();
        MDruntimes.add(0, 0);
        this.fitter = PolynomialCurveFitter.create(Ntdegree);

        this.reuse = true;
        this.opt = false;
        this.sample = true;
        this.knn = true;
    }

    public void extractData(RealMatrix records) {
        this.dataMatrix = records;
        this.N = records.getColumnDimension();
        this.M = records.getRowDimension();

        this.trainList = new ArrayList<>();
        this.allIndicesN = makeIntList(N);//new int[N];
        this.allIndicesM = makeIntList(M);//new int[M];
    }

    public void extractData(RealMatrix records, Boolean limit) {
        this.dataMatrix = records;
        this.N = records.getColumnDimension();
        this.M = records.getRowDimension();

        if (limit) {
            this.N = (this.N / 20) * 20;
            this.dataMatrix = records.getSubMatrix(0, this.M - 1, 0, this.N - 1);
        }

        this.trainList = new ArrayList<>();
        this.allIndicesN = makeIntList(N);//new int[N];
        this.allIndicesM = makeIntList(M);//new int[M];
    }

    public void preprocess() {
        //touch all of the data
        double touch = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                touch += this.dataMatrix.getEntry(i, j);
            }
        }
    }

    public Map<Integer, double[]> bundleMDTimeGuess() {
        Map<Integer, double[]> predVact = new HashMap<>();
        for (int Nt : trainTimeList.keySet()) {
            predVact.put(Nt, new double[]{trainTimeList.get(Nt), predictedTrainTimeList.getOrDefault(Nt, 0.0)});
        }
        return predVact;
    }

    // this is always called before anything else happens that iter
    public int computeNt(int iter, int currNt) {
        //if sampling is disabled, return M, then M+1
        if (!sample) {
            if (iter == 0) {
                return M;
            }
            return M + 1;
        }


        // tentative next Nt. If work reuse, go constant schedule. If not, increasing schedule
        // all "evenly spaced" for now, either 1% or 10% depending on data size
        int nextNt;
        nextNt = getNextNtFixedInterval(iter, currNt);


        //iter 0 is special because currNt has not been run yet, so no #s exist
        if (iter == 0) {
            NtList.add(nextNt);
            return nextNt;
        }

        if (nextNt > M) {
            return nextNt;
        }

        //for all other iters, MD has been run with currNt

        //if optimization mode set to on, run progress estimation
        if (opt) {
            nextNt = runProgressEstimation(iter, currNt, nextNt);
        //if not, run until K plateaus
        } else if (avgKDiff < 1){
            nextNt = M + 1;
        }
        NtList.add(nextNt);
        return nextNt;
    }

    //returns a constant value for number of data points sampled--either 10% or 1%
    public int getNextNtConstantSchedule() {
        double frac = (M < 10000) ? 0.10 : 0.01;
	    Double nextNt = frac * M;
        return nextNt.intValue();
    }

    //use this for increasing data. Evenly spaced, increasingly sized samples
    public int getNextNtFixedInterval(int iter, int currNt) {
        double frac = (M < 10000) ? 0.10 : 0.01;
        //frac = 0.10;
        int NtInterval = Math.max(10, new Double(frac * M).intValue());
        if (iter == 0) {
            return NtInterval;
        }
        return NtInterval + currNt;
    }

    public int getNextNtFixedSchedule(int iter, int currNt) {
        double[] NtSchedule = new double[]{0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, .30, .65, 1.5};//, 1.0, 1.5};
        Double Nt = NtSchedule[iter] * M;
        return Nt.intValue();
    }


    public double costModel(int k, boolean knn){
        if (knn) {
            //Cost model from KNN java run: a*N*(M^b) + c; a = 4.88383517848e-07 b = 2.09996767514 c = 425.760390427 3:1
            // 2.01532220508e-05 1.26883984061 -22.1487570101 for 1:1
            return Math.max(0.0, 2.01532220508e-05 * k * (Math.pow(M, 1.26883984061)) - 22.1487570101);

        } else {
            //Cost model from XMeans java run: a*N*(M^b) + c; a = 0.00124454969134 b = 1.25382761078 c = -603.399528341  3:1
            //2.21007892153e-05 1.34632909149 -21.6030357058 1:1
            return Math.max(0.0, 2.21007892153e-05 * k * (Math.pow(M, 1.34632909149 )) - 21.6030357058);
        }
    }

    public int runProgressEstimation(int iter, int currNt, int nextNt){
        //M*(lastk^scaling) + MDtime(currNt). Compute and store both predicted and actual
        //changing the objective to being of global time changes this check to be f(kt) + MD(t) < f(k_{t-1}) + 0
        // storing funck-k side as predicted, runtime pred as true.
        double NtTimeGuess = NtTimePredictOneStepGradient(iter, nextNt);

        double prevFk =  costModel(KList.get(currNt), knn);
        int kGuess = predictK(iter, nextNt); //iter needed for currNt ans one before
        double predFk =  costModel(kGuess, knn);

        trueObjective.put(currNt, prevFk + dropTimeSoFar);
        predictedObjective.put(nextNt, predFk + NtTimeGuess + dropTimeSoFar);

        fObjective.put(currNt, prevFk - predFk);
        dObjective.put(nextNt, NtTimeGuess);

        //keep going until (1) obj function say so or (2) not feasible yet
        if ((prevFk - predFk >= NtTimeGuess) || (!this.feasible) || iter < 2){ //(nextNt <= 1000){ //
            return nextNt;
        }
        return M+1;

    }

    //TODO: check indices here
    public double NtTimePredictOneStepGradient(int iter, int nextNt){
        if (iter == 1){
            double guess = MDDiff + prevMDTime;
            this.predictedTrainTimeList.put(nextNt, guess);
            return guess;
        }
        double ratio = MDDiff/ (NtList.get(iter-1) - NtList.get(iter-2));
        double guess = Math.max(0.0, prevMDTime + ratio*(nextNt - NtList.get(iter-1)));
        this.predictedTrainTimeList.put(nextNt, guess);
        return guess;
    }

    //predicting K for the "next" iteration and Nt
    public int predictK(int iter, int nextNt){
        if (iter == 1){
            int guess = kDiff + prevK;
            this.kPredList.put(nextNt, guess);
            return guess;
        }
        double ratio = (double) kDiff/ (NtList.get(iter-1) - NtList.get(iter-2));
        int guess = Math.max(0, prevK + (int) Math.round(ratio*(nextNt - NtList.get(iter-1))));
        this.kPredList.put(nextNt, guess);
        return guess;
    }

    public void updateMDRuntime(int iter, int currNt, double MDtime){
        dropTimeSoFar += MDtime;
        MDruntimes.add(currNt, MDtime);
        trainTimeList.put(currNt, MDtime);

        MDDiff = MDtime - prevMDTime;
        prevMDTime = MDtime;
    }


    public double[] evalK(double LBRThresh, RealMatrix currTransform, double constant) {
        double[] CI = new double[]{0, 0, 0};
        double q = 1.96;
        double prevMean = 0;
        //If this ever gets this big we're in big trouble, but it never does.
        long numPairs = Math.max((this.M) * ((this.M) - 1) / 2, (long) Integer.MAX_VALUE);
        int currPairs = Math.min(this.M, 500);//Math.max(5, this.M);//new Double(0.005*numPairs).intValue());
        while (currPairs < numPairs) {
            //log.debug("num pairs {}", currPairs);do
            CI = this.LBRCI(currTransform, currPairs, q, constant);
            //all stopping conditions here:  LB > wanted; UB < wanted; mean didn't change much from last time
            if ((CI[0] > LBRThresh) || (CI[2] < LBRThresh) || (Math.abs(CI[1] - prevMean) < .001)) {
                return CI;//LBRThresh;
            } else {
                currPairs *= 2;
                prevMean = CI[1];
            }
            if (currPairs < 0){
                System.exit(1);
            }
        }
        return CI;
    }


    public double[] LBRCI(RealMatrix transformedData, int numPairs, double threshold, double constant) {
        int K = transformedData.getColumnDimension();
        //int currNt = NtList.get(iter);

        int[][] indicesAB = randPairs(numPairs, M);
        int[] indicesA = indicesAB[0];
        int[] indicesB = indicesAB[1];
        int[] kIndices;

        RealVector transformedDists;
        RealVector trueDists;

        List<Double> LBRs;

        kIndices = Arrays.copyOf(allIndicesN,K);

        transformedDists = calcDistances(transformedData.getSubMatrix(indicesA,kIndices), transformedData.getSubMatrix(indicesB, kIndices)).mapMultiply(Math.sqrt(constant));
        trueDists = calcDistances(this.dataMatrix.getSubMatrix(indicesA,allIndicesN), this.dataMatrix.getSubMatrix(indicesB,allIndicesN));
        LBRs = calcLBRList(trueDists, transformedDists);
        return confInterval(LBRs, threshold);
    }


    //calls other function with constant set to 1
    public double[] LBRCI(RealMatrix transformedData, int numPairs, double threshold){
       return LBRCI(transformedData, numPairs, threshold, 1.0);
    }

    public int getN(){ return N;}

    public int getM(){return M;}

    ///TODO: do somethiing with the first runner information. Maybe just move this to PCAskiing and do this.feasible. Right now this only auto quits with objective function if you didn't improve
    public void setKDiff(int currK, int iter){
        kDiff = currK - prevK;
        if (kDiff <= 0){
            this.firstKDrop = false;
        }
        prevK = currK;
        kDiffList[iter % 3] = Math.abs(kDiff);
        if (iter >= 3) {
            avgKDiff = (kDiffList[0] + kDiffList[1] + kDiffList[2])/3.;
        }
    }

    public void addNtList(int Nt){ NtList.add(Nt); }

    public void setKList(int k, int v){ KList.put(k,v); }

    public void setLBRList(int k, double[] v){
        LBRList.put(k, v);
    }

    public void setTrainTimeList(int k, double v){
        trainTimeList.put(k, v);
    }

    public double[] getCurrKCI(){ return currKCI; }

    public int getNtList(int iter){ return NtList.get(iter); }

    public Map getLBRList(){ return LBRList; }

    public Map getTrainTimeList(){ return trainTimeList; }

    public Map getPredictedTrainTimeList(){ return predictedTrainTimeList; }

    public Map getKList(){ return KList; }

    public Map getKPredList(){ return kPredList; }

    public RealMatrix getDataMatrix(){ return dataMatrix; }

    public Map getTrueObjective() { return trueObjective; }

    public Map getPredictedObjective() { return predictedObjective; }

    public Map getfObjectiveCheck() { return fObjective; }

    public Map getdObjectiveCheck() { return dObjective; }

    public abstract void fit(int Nt);

    public abstract RealMatrix transform(int K);

}
