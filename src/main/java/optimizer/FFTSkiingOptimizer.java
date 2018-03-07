package optimizer;

import com.google.common.base.Stopwatch;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.jtransforms.fft.DoubleFFT_1D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class FFTSkiingOptimizer extends SkiingOptimizer {
    private static final Logger log = LoggerFactory.getLogger(FFTSkiingOptimizer.class);
    protected Map<Integer, Integer> KItersList;

    protected DoubleFFT_1D t;

    protected RealMatrix paddedInput;
    protected double[][] td;

    RealMatrix transformedData;
    long transformTime;

    public FFTSkiingOptimizer(double qThresh) {
        super(qThresh);
        this.KItersList = new HashMap<>();
    }

    @Override
    public void fit(int Nt) {
        paddedInput = new Array2DRowRealMatrix(this.M, 2*N);
        paddedInput.setSubMatrix(this.dataMatrix.getData(), 0, 0);
        td = paddedInput.getData();
        t  = new DoubleFFT_1D(N);

        for (int i = 0; i < this.M; i++) {
            t.realForwardFull(td[i]);
        }
        transformedData = new Array2DRowRealMatrix(td);
    }

    @Override
    //K must be even.
    public RealMatrix transform(int K) {
        //assert (K % 2 == 0);
        return transformedData.getSubMatrix(0,M-1,0,K-1);
    }

    public double[][] computeTransform(double targetLBR) {
        double[] LBR;
        int high = 2*N;
        int low = 0;
        int mid = (high + low) / 2;
        Stopwatch sw = Stopwatch.createUnstarted();
        sw.start();
        this.fit(M);

        //Binary search for lowest K that achieves LBR
        while (low != high) {
            LBR = evalK(targetLBR, this.transform(mid), 2. / N);
            if (LBR[0] <= targetLBR) {
                low = mid + 1;
            } else {
                high = mid;
            }
            mid = (low + high) / 2;
        }
        currKCI = evalK(targetLBR, this.transform(mid), 2. / N);
        sw.stop();
        transformTime = sw.elapsed(TimeUnit.MILLISECONDS);
        return this.transform(mid).getData();
    }


    public Map<String,Map<Integer, Double>> computeLBRs(){
        //confidence interval based method for getting K
        Map<Integer, Double> LBRs = new HashMap<>();
        Map<Integer, Double> times = new HashMap<>();
        Map<String, Map<Integer, Double>> results = new HashMap<>();

        Stopwatch sw =  Stopwatch.createUnstarted();

        sw.start();
        this.fit(M);
        sw.stop();
        times.put(0, (double) sw.elapsed(TimeUnit.MILLISECONDS));

        double[] CI;
        int interval = Math.max(2,this.N/30 + ((this.N/30) % 2)); //ensure even k always
        RealMatrix currTransform;
        for (int i = 2;i <= N; i+= interval){
            sw.reset();
            sw.start();
            currTransform = this.transform(i);
            sw.stop();

            CI = this.LBRCI(currTransform, M, qThresh, 2./N);
            log.debug("With K {}, LBR {} {} {}", i, CI[0], CI[1],CI[2]);
            LBRs.put(i, CI[1]);
            times.put(i, (double) sw.elapsed(TimeUnit.MILLISECONDS));
        }
        results.put("LBR", LBRs);
        results.put("time", times);
        return results;
    }

    public int getNextNt(int iter, int currNt) {
        return this.M;
    }

    public Map getKItersList(){ return KItersList; }

    public long getTransformTime() {
        return transformTime;
    }

}
