package runner;

import com.google.common.base.Stopwatch;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import stats.PCASVD;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static utils.util.stringToDoubleArray;

public class PCARunner {
    private static final Logger log = LoggerFactory.getLogger(PCARunner.class);

    double[][] finalTransform;
    double[] currLBR;
    int currNt;
    int iter;

    RealMatrix transformedData;
    int currK;
    PCASkiingOptimizer pcaOpt;
    Stopwatch sw;
    Stopwatch MD;

    Map<Long, Double> LBRProgress;
    Map<Long, Integer> KProgress;

    double qThresh;
    double lbr;

    int fixK;

    PCASkiingOptimizer.PCAAlgo algo;


    public PCARunner(double qThresh, double lbr, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, PCASkiingOptimizer.optimize optimize, PCASkiingOptimizer.sampling sample){
        this.algo = algo;
        pcaOpt = new PCASkiingOptimizer(qThresh, algo, reuse, optimize, sample);

        MD = Stopwatch.createUnstarted();
        sw = Stopwatch.createUnstarted();

        LBRProgress = new HashMap<>();
        KProgress = new HashMap<>();

        this.qThresh = qThresh;
        this.lbr = lbr;
    }

    public PCARunner(double qThresh, double lbr, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, PCASkiingOptimizer.optimize optimize, PCASkiingOptimizer.sampling sample, boolean knn){
        this.algo = algo;
        pcaOpt = new PCASkiingOptimizer(qThresh, algo, reuse, optimize, sample, knn);

        MD = Stopwatch.createUnstarted();
        sw = Stopwatch.createUnstarted();

        LBRProgress = new HashMap<>();
        KProgress = new HashMap<>();

        this.qThresh = qThresh;
        this.lbr = lbr;
    }

    public PCARunner(double qThresh, double lbr, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, PCASkiingOptimizer.optimize optimize){
        this.algo = algo;
        pcaOpt = new PCASkiingOptimizer(qThresh, algo, reuse, optimize, PCASkiingOptimizer.sampling.SAMPLE);

        MD = Stopwatch.createUnstarted();
        sw = Stopwatch.createUnstarted();

        LBRProgress = new HashMap<>();
        KProgress = new HashMap<>();

        this.qThresh = qThresh;
        this.lbr = lbr;
    }

    //for baseline.
    public PCARunner(double qThresh, double lbr, PCASkiingOptimizer.PCAAlgo algo){
        this.algo = algo;
        pcaOpt = new PCASkiingOptimizer(qThresh, algo);

        MD = Stopwatch.createUnstarted();
        sw = Stopwatch.createUnstarted();

        LBRProgress = new HashMap<>();
        KProgress = new HashMap<>();

        this.qThresh = qThresh;
        this.lbr = lbr;
    }

    //for work reuse debug. K is fixed here, Nt is as well, and we aren't trying to hit an LBR
    public PCARunner(double qThresh, int k, PCASkiingOptimizer.PCAAlgo algo) {
        this.algo = algo;
        fixK = k;
        pcaOpt = new PCASkiingOptimizer(qThresh, algo);

        MD = Stopwatch.createUnstarted();
        sw = Stopwatch.createUnstarted();
        LBRProgress = new HashMap<>();
        this.qThresh = qThresh;
    }

    public void consume(RealMatrix records) throws Exception {
        iter = 0;
        currNt = 0;

        pcaOpt.extractData(records);
        System.out.println(String.format("Extracted %d Records of dim %d", pcaOpt.getM(),pcaOpt.getN()));
        pcaOpt.preprocess();
        currNt = pcaOpt.computeNt(iter, currNt);
        pcaOpt.warmUp(currNt);
	    sw.start();
        do {
            sw.stop();
            System.gc();
            sw.start();
            MD.reset();
            MD.start();
            pcaOpt.fit(currNt);
            currK = pcaOpt.getKCI(currNt, lbr); //function to get knee for K for this transform;
            MD.stop();

            //store how long (MD) this currNt took, diff between last and this, and store this as prevMD
            pcaOpt.updateMDRuntime(iter, currNt, (double) MD.elapsed(TimeUnit.MILLISECONDS));

            //returns the LBR CI from getKCI and then store it
            currLBR = pcaOpt.getCurrKCI();
            pcaOpt.setLBRList(currNt, currLBR);

            //store the K obtained and the diff in K from this currNt
            pcaOpt.setKList(currNt, currK);
            pcaOpt.setKDiff(currK, iter);

            LBRProgress.put(sw.elapsed(TimeUnit.MILLISECONDS), currLBR[1]);
            KProgress.put(sw.elapsed(TimeUnit.MILLISECONDS), currK);
            currNt = pcaOpt.computeNt(++iter, currNt);
        } while ((currNt <= pcaOpt.getM()) && (iter < 30));
        sw.stop();
	    transformedData = pcaOpt.transform(currK);

        finalTransform = transformedData.getData();
    }

    // runs "MD" with a constant number of sampled data points always, and a fixed K. Just returns whatever LBR it finds
    public List<Double>[] debug(RealMatrix records) {
        List<Double> MDruntime = new ArrayList<>();
        List<Double> LBRList = new ArrayList<>();
        iter = 0;

        pcaOpt.extractData(records);
        System.out.println(String.format("Extracted %d Records of dim %d", pcaOpt.getM(),pcaOpt.getN()));
        pcaOpt.preprocess();
        currNt = pcaOpt.getNextNtConstantSchedule(); //should account for previous LBRs seen. for now just stop after 20
        do {
            System.gc();
            MD.reset();
            MD.start();
            pcaOpt.fit(currNt);
            currK = pcaOpt.getKCIFixed(fixK);
            MD.stop();

            currLBR = pcaOpt.getCurrKCI();

            MDruntime.add((double) MD.elapsed(TimeUnit.MILLISECONDS));
            LBRList.add(currLBR[1]);
            LBRProgress.put(sw.elapsed(TimeUnit.MILLISECONDS), currLBR[1]);
            iter++;
        } while (iter < 25);
        return new List[]{MDruntime, LBRList};
    }

    public Map<String,Map<Integer, Double>> genBasePlots(RealMatrix records){
        pcaOpt = new PCASkiingOptimizer(qThresh, algo);
        pcaOpt.extractData(records, Boolean.TRUE);
        pcaOpt.preprocess();

        return pcaOpt.computeLBRs();
    }

    public double[] getDataSpectrum(RealMatrix records){
        pcaOpt = new PCASkiingOptimizer(qThresh, PCASkiingOptimizer.PCAAlgo.SVD);
        pcaOpt.extractData(records);
        PCASVD svd = new PCASVD(pcaOpt.getDataMatrix());
        return svd.getSpectrum();
    }

    public long[] baselineSVD(RealMatrix records) {
        pcaOpt = new PCASkiingOptimizer(qThresh, algo);
        pcaOpt.extractData(records);
        pcaOpt.preprocess();

        long[] output = pcaOpt.getFullSVD(lbr);
        transformedData = pcaOpt.transform(new Long(output[0]).intValue());
        finalTransform = transformedData.getData();

        return output;
    }

    public long[] oracleSVD(RealMatrix records, double propn) {
        pcaOpt = new PCASkiingOptimizer(qThresh, algo);
        pcaOpt.extractData(records);
        pcaOpt.preprocess();

        long[] output = pcaOpt.getFullSVD(lbr, propn);
        transformedData = pcaOpt.transform(new Long(output[0]).intValue());
        finalTransform = transformedData.getData();

        return output;
    }

    //this is what you would want to dump out in the end to csv
    public double[][] getLabeledFinalTransform(String[] labels) {
        RealMatrix labeledTransform = new Array2DRowRealMatrix(transformedData.getRowDimension(), transformedData.getColumnDimension()+1);
        labeledTransform.setSubMatrix(finalTransform, 0, 1);
        labeledTransform.setColumn(0, stringToDoubleArray(labels));
        return labeledTransform.getData();
    }

    public double[][] getFinalTransform() {
        return finalTransform;
    }

    public long totalTime() { return sw.elapsed(TimeUnit.MILLISECONDS);}

    public int finalK() { return currK; }

    public int getM() { return pcaOpt.getM(); }

    public int getNt() { return pcaOpt.getNtList(iter-1); }

    public double[] getFinalLBR() { return pcaOpt.getCurrKCI(); }

    public Map<Integer, double[]> getLBR() { return pcaOpt.getLBRList();}

    public Map<Integer, Double> getTime(){
        return pcaOpt.getTrainTimeList();
    }

    public Map<Integer, Integer> getKList(){
        return pcaOpt.getKList();
    }

    public Map<Integer, double[]> getMDRuntimes() { return pcaOpt.bundleMDTimeGuess(); }

    public Map<Integer, Double> getTrueObjective() { return pcaOpt.getTrueObjective(); }

    public Map<Integer, Double> getPredictedObjective() { return pcaOpt.getPredictedObjective(); }

    public Map<Integer, Double> getfObjectiveCheck() { return pcaOpt.getfObjectiveCheck(); }

    public Map<Integer, Double> getdObjectiveCheck() { return pcaOpt.getdObjectiveCheck(); }

    public Map<Integer, Integer> getKItersList(){
        return pcaOpt.getKItersList();
    }

    public Map<Integer, Integer> getKPred() { return pcaOpt.getKPredList(); }

    public Map<Integer, Double> getPredTrainTimes() { return pcaOpt.getPredictedTrainTimeList(); }

    public Map<Integer, Double> getTrainTimes() { return pcaOpt.getTrainTimeList(); }

    public Map<Long, Double> getLBRProgress() {return LBRProgress; }

    public Map<Long, Integer> getKProgress() {return KProgress; }

    public int getNumIters(){
        return iter;}
}
