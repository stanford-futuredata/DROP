package experiments;


import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by meep_me on 9/1/16.
 */
public class WorkReuseExperiments extends Experiment {
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/ReuseExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String kAndRuntimeOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_%s_%s",minute.format(date),dataset, algo, lbr, qThresh, reuse, opt);
        return String.format(baseString + day.format(date) + "/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        Date date = new Date();
        int numTrials = 50;
        long tempRuntime;
        int tempK;

        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        double qThresh = Double.parseDouble(args[2]);
        int kExp = Integer.parseInt(args[3]);
        PCASkiingOptimizer.optimize opt =   PCASkiingOptimizer.optimize.valueOf(args[4]);
        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(qThresh);
        System.out.println(kExp);
        System.out.println(opt);


        PCASkiingOptimizer.PCAAlgo[] algos = {PCASkiingOptimizer.PCAAlgo.TROPP, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.work[] works = {PCASkiingOptimizer.work.REUSE, PCASkiingOptimizer.work.NOREUSE};


        Map<Integer, Long> finalk_runtime;

        RealMatrix data;

        for (PCASkiingOptimizer.PCAAlgo algo: algos) {
            for (PCASkiingOptimizer.work reuse: works) {
                //silly, but easier to remember if output file corresponds to each thing.
                finalk_runtime = new HashMap<>();
                data = getData(dataset);
                tempK = 0;
                tempRuntime = 0;

                for (int i = 0; i < numTrials; i++) {
                    PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
                    drop.consume(data);

                    tempK += drop.finalK();
                    tempRuntime += drop.totalTime();
                }
                //dumping k,runtime for each config
                finalk_runtime.put(tempK/numTrials, tempRuntime/numTrials);
                mapIntLongToCSV(finalk_runtime, kAndRuntimeOutFile(dataset,lbr,qThresh,algo,reuse,date,opt));
            }
        }

    }
}
