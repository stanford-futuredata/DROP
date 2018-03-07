package optimizer;

import com.google.common.base.Stopwatch;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class PAASkiingOptimizer extends SkiingOptimizer{
    private static final Logger log = LoggerFactory.getLogger(PAASkiingOptimizer.class);
    protected Map<Integer, Integer> KItersList;
    protected List<Integer> factors;

    long transformTime;

    public PAASkiingOptimizer(double qThresh){
        super(qThresh);
        this.KItersList = new HashMap<>();
    }

    @Override
    public void fit(int Nt) {
    }

    @Override
    public void preprocess() {
        super.preprocess();
        this.factors = this.findFactors();
    }


    @Override
    public RealMatrix transform(int K) {
        // Implementation of PAA. TODO: not optimized as Keogh says
        assert (this.N % K == 0 );
        RealMatrix output = new Array2DRowRealMatrix(this.M, K);
        RealVector currVec;
        double temp;
        int entriesAveraged = this.N / K;

        if (K == this.N){
            return this.dataMatrix;
        }

        for (int i = 0; i < this.M; i++){
            currVec = this.dataMatrix.getRowVector(i);
            temp = 0;
            for (int j = 0; j < this.N; j++){
                if (j % entriesAveraged == 0 && j != 0){
                    output.setEntry(i, j/entriesAveraged - 1,temp/entriesAveraged);
                    temp = 0;
                }
                temp += currVec.getEntry(j);
                if (j == this.N - 1){
                    output.setEntry(i,this.N/entriesAveraged - 1, temp/entriesAveraged);
                }
            }
        }
        return output;
    }


    private List<Integer> findFactors(){
        List<Integer> factors = new ArrayList<>();
        for (int i = 1; i <= this.N; i++) {
            if (this.N % i == 0) factors.add(i);
        }
        return factors;
    }

    public double[][] computeTransform(double targetLBR) {
        double[] LBR;
        int k;
        int high = factors.size();
        int low = 0;
        int mid = (high + low) / 2;
        Stopwatch sw = Stopwatch.createUnstarted();
        sw.start();

        //Binary search for lowest K that achieves LBR
        while (low != high) {
            k = factors.get(mid);
            LBR = evalK(targetLBR, this.transform(k), ((double) N) /k);
            if (LBR[0] <= targetLBR) {
                low = mid + 1;
            } else {
                high = mid;
            }
            mid = (low + high) / 2;
        }
        k = factors.get(mid);
        currKCI = evalK(targetLBR, this.transform(k), ((double) N) /k);
        sw.stop();
        transformTime = sw.elapsed(TimeUnit.MILLISECONDS);
        return this.transform(k).getData();
    }



    public Map<String,Map<Integer, Double>> computeLBRs(){
        Map<Integer, Double> LBRs = new HashMap<>();
        Map<Integer, Double> times = new HashMap<>();
        Map<String, Map<Integer, Double>> results = new HashMap<>();

        Stopwatch sw =  Stopwatch.createUnstarted();

        sw.start();
        this.fit(M);
        sw.stop();
        times.put(0, (double) sw.elapsed(TimeUnit.MILLISECONDS));

        double[] CI;
        RealMatrix currTransform; //= new Array2DRowRealMatrix();
        for (int i: factors){
            sw.reset();
            sw.start();
            currTransform = this.transform(i);
            sw.stop();

            CI = this.LBRCI(currTransform,M, qThresh, ((double) N) /i);
            log.debug("With K {}, LBR {} {} {}", i, CI[0], CI[1],CI[2]);
            LBRs.put(i, CI[1]);
            times.put(i, (double) sw.elapsed(TimeUnit.MILLISECONDS));
        }
        results.put("LBR", LBRs);
        results.put("time", times);
        return results;
    }

    public Map getKItersList(){ return KItersList; }

    public long getTransformTime() {
        return transformTime;
    }
}
