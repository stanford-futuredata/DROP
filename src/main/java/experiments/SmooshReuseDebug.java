package experiments;

import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;
import runner.PCARunner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import static utils.util.*;
/**
 * Created by sahaana on 12/7/17.
 */
public class SmooshReuseDebug extends Experiment {
    public static String baseString = "src/main/java/experiments/SmooshDebug/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String runtimeOutFile(Date date, String dataset, int K, PCASkiingOptimizer.PCAAlgo algo){
        String output = String.format("%s_%s_%s_k%d",minute.format(date),dataset, algo, K);
        return String.format(baseString + day.format(date) + "/runtime/%s.csv", output);
    }

    private static String LBROutFile(Date date, String dataset, int K, PCASkiingOptimizer.PCAAlgo algo){
        String output = String.format("%s_%s_%s_k%d",minute.format(date),dataset, algo, K);
        return String.format(baseString + day.format(date) + "/lbr/%s.csv", output);
    }

    //java -Xms8g ${JAVA_OPTS} -cp 'target/classes/' experiments.OjasBaselineComparison
    public static void main(String[] args) throws Exception {
        Date date = new Date();

        String dataset = args[0];

        int K = Integer.parseInt(args[1]);
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.valueOf(args[2]);
        System.out.println(dataset);
        System.out.println(K);
        System.out.println(algo);

        double qThresh = 1.96;

        RealMatrix data = getData(dataset);
        PCARunner drop = new PCARunner(qThresh, K, algo);

        List<Double>[] output = drop.debug(data);
        double[] MDRuntimes = doubleListToPrimitive(output[0]);
        double[] LBRs = doubleListToPrimitive(output[1]);

        doubleListToCSV(MDRuntimes, runtimeOutFile(date, dataset, K, algo));
        doubleListToCSV(LBRs, LBROutFile(date, dataset, K, algo));
    }
}
