package experiments;

import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;
import runner.OjasRunner;
import runner.PCARunner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by sahaana on 12/7/17.
 */
public class OjasBaselineComparison extends Experiment {
    public static String baseString = "src/main/java/experiments/SGDBaseline/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String ojasOutFile(Date date, String dataset, int batchSize, int iters, double learningRate, double freq, int K){
        String output = String.format("%s_%s_b%d_iter%d_eta%f_freq%g_k%d",minute.format(date),dataset, batchSize, iters, learningRate, freq, K);
        return String.format(baseString + day.format(date) + "/ojas/%s.csv", output);
    }

    private static String ojasConvergenceOutFile(Date date, String dataset, int batchSize, int iters, double learningRate, double freq, int K){
        String output = String.format("%s_%s_b%d_iter%d_eta%f_freq%g_k%d",minute.format(date),dataset, batchSize, iters, learningRate, freq, K);
        return String.format(baseString + day.format(date) + "/ojas_convergence/%s.csv", output);
    }

    private static String dropKOutFile(Date date, String dataset, double qThresh, double lbr, int kExp, PCASkiingOptimizer.work reuse, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_kexp%d_%s_%s",minute.format(date),dataset,lbr,qThresh,kExp,reuse,opt);
        return String.format(baseString + day.format(date) + "/drop_k/%s.csv", output);
    }

    private static String dropLBROutFile(Date date, String dataset, double qThresh, double lbr, int kExp, PCASkiingOptimizer.work reuse, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_kexp%d_%s_%s",minute.format(date),dataset,lbr,qThresh,kExp,reuse,opt);
        return String.format(baseString + day.format(date) + "/drop_lbr/%s.csv", output);
    }
    //java -Xms8g ${JAVA_OPTS} -cp 'target/classes/' experiments.OjasBaselineComparison
    public static void main(String[] args) throws Exception {
        Map<Long, Double> dropRuntime = new HashMap<>();

        Date date = new Date();

        String dataset = args[0];

        //Oja's Params
        int K = Integer.parseInt(args[1]);
        int freq = Integer.parseInt(args[2]);
        int iters = Integer.parseInt(args[3]);
        int batchSize = Integer.parseInt(args[4]);
        double learningRate = Double.parseDouble(args[5]);

        System.out.println(dataset);
        System.out.println(K);
        System.out.println(iters);
        System.out.println(batchSize);
        System.out.println(learningRate);

        //DROP Params
        double lbr = 0.98;
        double qThresh = 1.96;
        int kExp = 2;
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.TROPP;
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.NOREUSE;
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.OPTIMIZE;

        System.out.println(lbr);
        System.out.println(qThresh);
        System.out.println(kExp);
        System.out.println(algo);
        System.out.println(reuse);
        System.out.println(opt);



        RealMatrix data = getData(dataset);
        PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
        OjasRunner ojas = new OjasRunner(data, batchSize, iters, learningRate, freq, K);
        drop.consume(data);
        ojas.consume(data);

        mapLongDoubleToCSV(ojas.getRuntimeLBR(), ojasOutFile(date, dataset, batchSize, iters, learningRate, freq, K));
        mapLongDoubleToCSV(ojas.getRuntimeConvergence(), ojasConvergenceOutFile(date, dataset, batchSize, iters, learningRate, freq, K));
        mapLongDoubleToCSV(drop.getLBRProgress(), dropLBROutFile(date, dataset, qThresh, lbr, kExp, reuse, opt));
        mapLongIntToCSV(drop.getKProgress(), dropKOutFile(date, dataset, qThresh, lbr, kExp, reuse, opt));
    }
}
