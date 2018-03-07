package experiments;

import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created by meep_me on 9/1/16.
 */
public class SpectrumDump extends Experiment{
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/baselineExperiments/spectrum";

    private static String spectrumOutFile(String dataset){
        return String.format(baseString + "/%s_spec.csv", dataset);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        double qThresh = Double.parseDouble(args[2]);
        System.out.println(dataset);

        double[] spectrum;

        RealMatrix data = getData(dataset);

        PCARunner pcaDrop = new PCARunner(qThresh, lbr, PCASkiingOptimizer.PCAAlgo.SVD);
        spectrum = pcaDrop.getDataSpectrum(data);

        doubleListToCSV(spectrum, spectrumOutFile(dataset));
    }

}
