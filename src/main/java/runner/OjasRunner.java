package runner;

import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import stats.PCAOjas;

import java.util.Map;

public class OjasRunner {
    private static final Logger log = LoggerFactory.getLogger(OjasRunner.class);

    PCAOjas oja;
    int K;

    public OjasRunner(RealMatrix data, int batchSize, int iters, double learningRate, int freq, int K){
        oja = new PCAOjas(data, batchSize, iters, learningRate, freq);
        this.K = K;
    }

    public void consume(RealMatrix dataToTransform) throws Exception {
        oja.transform(dataToTransform, K);
    }

    public Map<Long, Double> getRuntimeLBR()
    {
        return oja.getLBRProgress();
    }

    public Map<Long, Double> getRuntimeConvergence(){
        return oja.getConvergenceProgress();
    }

}
