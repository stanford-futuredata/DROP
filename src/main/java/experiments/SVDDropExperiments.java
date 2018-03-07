package experiments;

import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by meep_me on 9/1/16.
 */
public class SVDDropExperiments extends Experiment {
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/SVDExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");

    private static String kOutFile(String dataset, double lbr, double qThresh, int kExp, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_kexp%d_%s_%s",minute.format(date),dataset,lbr,qThresh,kExp,reuse, opt);
        return String.format(baseString + day.format(date) + "/NTvK/%s.csv", output);
    }

    private static String timeEstimateOutFile(String dataset, double lbr, double qThresh, int kExp, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_kexp%d_%s_%s",minute.format(date),dataset,lbr,qThresh,kExp,reuse,opt);
        return String.format(baseString + day.format(date) + "/MDtimeEst/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        Date date = new Date();
        int numTrials = 50;
        double[] defDub = {0,0};


        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        double qThresh = Double.parseDouble(args[2]);
        int kExp = Integer.parseInt(args[3]);
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.valueOf(args[4]);
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.valueOf(args[5]);
        PCASkiingOptimizer.optimize opt =   PCASkiingOptimizer.optimize.valueOf(args[6]);

        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(qThresh);
        System.out.println(kExp);
        System.out.println(algo);
        System.out.println(reuse);
        System.out.println(opt);


        Map<Integer, Double> kcounts = new HashMap<>();
        Map<Integer, Double> tcounts = new HashMap<>();
        Map<Integer, Integer> kResults = new HashMap<>();
        Map<Integer, double[]> MDTimeResults = new HashMap<>();

        Map<Integer, Integer> tempkResults;
        Map<Integer, double[]> tempMDTimeResults;


        RealMatrix data = getData(dataset);

        for (int i = 0; i < numTrials; i++){
            PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
            drop.consume(data);

            tempkResults = drop.getKList();
            tempMDTimeResults = drop.getMDRuntimes();

            //update K results
            for (Map.Entry<Integer, Integer> entry: tempkResults.entrySet()) {
                int key = entry.getKey();
                int kval = entry.getValue();

                kcounts.put(key, 1 + kcounts.getOrDefault(key,0.0));
                kResults.put(key, kval + kResults.getOrDefault(key,0));
            }

            //update time results
            for (Map.Entry<Integer, double[]> entry: tempMDTimeResults.entrySet()) {
                int key = entry.getKey();
                double treal = entry.getValue()[0];
                double tguess = entry.getValue()[1];
                double oreal = MDTimeResults.getOrDefault(key,defDub)[0];
                double oguess = MDTimeResults.getOrDefault(key,defDub)[1];

                tcounts.put(key, 1 + tcounts.getOrDefault(key,0.0));
                MDTimeResults.put(key, new double[] {treal+oreal, tguess+oguess});
            }
        }

        kResults = scaleIntMap(kResults,kcounts);
        MDTimeResults = scaleDouble2Map(MDTimeResults, tcounts);

        mapIntToCSV(kResults, kOutFile(dataset,lbr,qThresh,kExp,reuse,date,opt));
        mapDouble2ToCSV(MDTimeResults, timeEstimateOutFile(dataset,lbr,qThresh,kExp,reuse,date,opt));
    }

}
