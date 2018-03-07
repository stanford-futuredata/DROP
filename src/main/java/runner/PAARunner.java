package runner;

import com.google.common.base.Stopwatch;
import optimizer.PAASkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

import static utils.util.stringToDoubleArray;

public class PAARunner {
    private static final Logger log = LoggerFactory.getLogger(PAARunner.class);
    double[][] finalTransform;
    int currK;
    int currNt;
    int iter;
    long transformTime;

    PAASkiingOptimizer paaOpt;
    Stopwatch sw;

    Map<String, Long> times;

    double lbr;

    public PAARunner(double qThresh, double lbr){
        iter = 0;
        currNt = 0;
        paaOpt = new PAASkiingOptimizer(qThresh);
        sw = Stopwatch.createUnstarted();

        times = new HashMap<>();

        this.lbr = lbr;
    }

    public void consume(RealMatrix records) throws Exception {
        paaOpt.extractData(records);
        paaOpt.preprocess();
        finalTransform = paaOpt.computeTransform(lbr);
        currK = finalTransform[0].length;
        transformTime = paaOpt.getTransformTime();
    }

    public Map<String,Map<Integer, Double>> genBasePlots(RealMatrix records){
        paaOpt.extractData(records, Boolean.TRUE);
        log.debug("Extracted {} Records of len {}", paaOpt.getM(), paaOpt.getN());
        paaOpt.preprocess();
        log.debug("Processed Data");

        log.debug("Beginning PAA base run");
        return paaOpt.computeLBRs();
    }


    public long totalTime() {
        return transformTime;
    }

    public int finalK() {
        return currK;
    }

    //this is what you would want to dump out in the end to csv
    public double[][] getLabeledFinalTransform(String[] labels) {
        RealMatrix labeledTransform = new Array2DRowRealMatrix(finalTransform.length, finalTransform[0].length + 1);
        labeledTransform.setSubMatrix(finalTransform, 0, 1);
        labeledTransform.setColumn(0, stringToDoubleArray(labels));
        return labeledTransform.getData();
    }

    public double[][] getFinalTransform() {
        return finalTransform;
    }

    public Map<Integer, double[]> getLBR() { return paaOpt.getLBRList();}

    public Map<Integer, Double> getTime(){
        return paaOpt.getTrainTimeList();
    }

    public Map<Integer, Integer> getKList(){
        return paaOpt.getKList();
    }

    public Map<Integer, Integer> getKItersList(){
        return paaOpt.getKItersList();
    }

}
