package runner;

import com.google.common.base.Stopwatch;
import optimizer.FFTSkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

import static utils.util.stringToDoubleArray;

public class FFTRunner {
    private static final Logger log = LoggerFactory.getLogger(FFTRunner.class);
    double[][] finalTransform;
    int currK;
    int currNt;
    int iter;
    long transformTime;

    FFTSkiingOptimizer fftOpt;
    Stopwatch sw;

    Map<String, Long> times;

    double lbr;

    public FFTRunner(double qThresh, double lbr){
        iter = 0;
        currNt = 0;
        fftOpt = new FFTSkiingOptimizer(qThresh);
        sw = Stopwatch.createUnstarted();

        times = new HashMap<>();

        this.lbr = lbr;

    }

    public void consume(RealMatrix records) throws Exception {
        fftOpt.extractData(records);
        fftOpt.preprocess();
        finalTransform = fftOpt.computeTransform(lbr);
        currK = finalTransform[0].length;
        transformTime = fftOpt.getTransformTime();
    }


    public Map<String,Map<Integer, Double>> genBasePlots(RealMatrix records){
        fftOpt.extractData(records, Boolean.TRUE);
        log.debug("Extracted {} Records of len {}", fftOpt.getM(), fftOpt.getN());
        fftOpt.preprocess();
        log.debug("Processed Data");

        log.debug("Beginning FFT base run");
        return fftOpt.computeLBRs();

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

    public Map<Integer, double[]> getLBR() { return fftOpt.getLBRList();}

    public Map<Integer, Double> getTime(){
        return fftOpt.getTrainTimeList();
    }

    public Map<Integer, Integer> getKList(){
        return fftOpt.getKList();
    }

    public Map<Integer, Integer> getKItersList(){
        return fftOpt.getKItersList();
    }

}
