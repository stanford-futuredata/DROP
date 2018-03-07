package experiments;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;
import utils.CSVTools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by meep_me on 9/1/16.
 */
public class OracleBinSearchExperiments extends Experiment {
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/OracleExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String kAndRuntimeOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, double propn, Date date){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_p%.4f",minute.format(date),dataset, algo, lbr, qThresh, propn);
        return String.format(baseString + day.format(date) + "/KR/%s.csv", output);
    }

    private static String lbrOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, double propn, Date date){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_p%.4f",minute.format(date),dataset, algo, lbr, qThresh, propn);
        return String.format(baseString + day.format(date) + "/lbr/%s.csv", output);
    }

    private static String transformOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, Date date){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f",minute.format(date),dataset, algo, lbr, qThresh);
        return String.format("/lfs/raiders6/0/sahaana/data/MIC_DROP/oracle/"+day.format(date)+"/%s.csv", output);
    }

    //todo: spit out lbr too

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        Date date = new Date();
        long[] tempKRuntime;
        long tempRuntime;
        int tempK;

        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        PCASkiingOptimizer.PCAAlgo algo =  PCASkiingOptimizer.PCAAlgo.valueOf(args[2]);
        double propn = Double.parseDouble(args[3]);
        int numTrials = Integer.parseInt(args[4]);
        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(algo);
        System.out.println(propn);
        System.out.println(numTrials);

        double qThresh = 1.96;
        Map<Integer, Long> Kruntimes = new HashMap<>();
        double[] finalLBR = new double[]{0, 0, 0};

        //data = getData(dataset);
        CSVTools.DataLabel d = getLabeledData(dataset);
        RealMatrix data = new Array2DRowRealMatrix(d.matrix);
        String[] labels = d.origLabels;

        tempK = 0;
        tempRuntime = 0;

        for (int i = 0; i < numTrials; i++){
            PCARunner drop = new PCARunner(qThresh, lbr, algo);
            tempKRuntime = drop.oracleSVD(data, propn);
            tempK += tempKRuntime[0];
            tempRuntime += tempKRuntime[1];
            double[] templbr = drop.getFinalLBR();
            for (int ii = 0; ii < 3; ii++) {
                finalLBR[ii] += templbr[ii];
            }

            //dump out an arbitrary low dimensional representation at end of DROP
            if (i == numTrials-1){
                double2dListToCSV(drop.getLabeledFinalTransform(labels), transformOutFile(dataset,lbr,qThresh,algo,date));
            }
        }

        //this is dumb //why? //because I put it into a map just so I can print it
        Kruntimes.put(tempK / numTrials, tempRuntime / numTrials);
        for (int ii = 0; ii < 3; ii++) {
            finalLBR[ii] = finalLBR[ii]/numTrials;
        }

        mapIntLongToCSV(Kruntimes, kAndRuntimeOutFile(dataset, lbr, qThresh, algo, propn, date));
        doubleListToCSV(finalLBR, lbrOutFile(dataset, lbr, qThresh, algo, propn, date));
    }
}
