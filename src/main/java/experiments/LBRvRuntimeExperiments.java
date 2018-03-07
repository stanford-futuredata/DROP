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
public class LBRvRuntimeExperiments extends Experiment {
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/LBRvTimeExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String timeOutFile(String dataset, double qThresh, int kExp, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_%s_q%.2f_kexp%d_%s_%s",minute.format(date),dataset, algo, qThresh, kExp, reuse,opt);
        return String.format(baseString + day.format(date) + "/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        Date date = new Date();
        int numTrials = 50;
        long tempRuntime;

        String dataset = args[0];
        double qThresh = Double.parseDouble(args[1]);
        int kExp = Integer.parseInt(args[2]);
        PCASkiingOptimizer.optimize opt =   PCASkiingOptimizer.optimize.valueOf(args[3]);
        System.out.println(dataset);
        System.out.println(qThresh);
        System.out.println(kExp);
        System.out.println(opt);

        double[] lbrs = {0.80,.85,0.9,0.95,0.98};
        PCASkiingOptimizer.PCAAlgo[] algos = {PCASkiingOptimizer.PCAAlgo.SVD, PCASkiingOptimizer.PCAAlgo.TROPP, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.work[] options = {PCASkiingOptimizer.work.NOREUSE, PCASkiingOptimizer.work.REUSE};
        Map<Double, Long> runtimes; //lbr > runtime for all configs

        RealMatrix data = getData(dataset);

        for (PCASkiingOptimizer.PCAAlgo algo: algos){
            for (PCASkiingOptimizer.work reuse: options){
                runtimes = new HashMap<>();
                for (double lbr: lbrs){
                    tempRuntime = 0;
                    for (int i = 0; i < numTrials; i++){
                        PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
                        drop.consume(data);
                        tempRuntime += drop.totalTime();
                    }
                    runtimes.put(lbr,tempRuntime/numTrials);
                }
                mapDoubleLongToCSV(runtimes, timeOutFile(dataset,qThresh,kExp,algo,reuse,date,opt));
            }
        }
    }
}
