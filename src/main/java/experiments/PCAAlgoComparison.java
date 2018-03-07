package experiments;

import com.google.common.base.Stopwatch;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import runner.PCARunner;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.*;
import utils.CSVTools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;

/**
 * Created by meep_me on 9/1/16.
 */
public class PCAAlgoComparison extends Experiment {
    public static String baseString = "src/main/java/experiments/PCAExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");

    private static String allOutFile(String dataset, Date date){
        String output = String.format("%s_%s",minute.format(date),dataset);
        return String.format(baseString + day.format(date) + "/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception {
        Date date = new Date();
        String dataset = "StarLightCurves";
        int k = 35;
        //Input arguments: data, LBR, numtrials, and opt or not
        int numTrials = Integer.parseInt(args[0]);
        System.out.println(dataset);
        System.out.println(numTrials);

        //Fixed arguments:
        Stopwatch sw = Stopwatch.createUnstarted();
        RealMatrix data;
        String labels[];
        double[] pcaSVD = new double[numTrials];
        double[] pcaSMILE = new double[numTrials];
        double[] pcaOjas = new double[numTrials];
        double[] pcaP = new double[numTrials];
        double[] pcaTropp = new double[numTrials];
        PCASVD svd;
        PCASmile smile;
        PCAOjas ojas;
        PPCASmile ppca;
        PCATropp tropp;





        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);

        //just to warm up
        tropp = new PCATropp(data);
        tropp.transform(data, k);


        for (int i = 0; i < numTrials; i++) {
            System.out.println(i);
            sw.start();
            svd = new PCASVD(data);
            svd.transform(data, k);
            sw.stop();
            pcaSVD[i] = (double) sw.elapsed(TimeUnit.MILLISECONDS);
            sw.reset();

            sw.start();
            smile = new PCASmile(data);
            smile.transformOld(data, k);
            sw.stop();
            pcaSMILE[i] = (double) sw.elapsed(TimeUnit.MILLISECONDS);
            sw.reset();

            sw.start();
            ojas = new PCAOjas(data, 100, 2000, 1e-5, 100000000);
            ojas.transform(data, k);
            sw.stop();
            pcaOjas[i] = (double) sw.elapsed(TimeUnit.MILLISECONDS);
            sw.reset();

            sw.start();
            ppca = new PPCASmile(data);
            ppca.transform(data, k);
            sw.stop();
            pcaP[i] = (double) sw.elapsed(TimeUnit.MILLISECONDS);
            sw.reset();

            sw.start();
            tropp = new PCATropp(data);
            tropp.transform(data, k);
            sw.stop();
            pcaTropp[i] = (double) sw.elapsed(TimeUnit.MILLISECONDS);
            sw.reset();

        }

        double[][] outputValues = new double[][]{pcaSVD, pcaSMILE, pcaOjas, pcaP, pcaTropp};
        double2dListToCSV(outputValues, allOutFile(dataset, date));
    }
}
