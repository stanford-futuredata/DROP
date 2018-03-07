package experiments;

import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.KMeans;
import stats.KNN;
import stats.XMeans;
import utils.CSVTools.DataLabel;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by sahaana on 12/7/17.
 */
public class XMeansRuntimeModeling extends Experiment {
    public static String baseString = "src/main/java/experiments/XMeansCostModel/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String runtimeOutFile(Date date, String dataset){
        String output = String.format("%s_%s",minute.format(date),dataset);
        return String.format(baseString + day.format(date) + "/%s.csv", output);
    }

    //java -Xms8g ${JAVA_OPTS} -cp 'target/classes/' experiments.OjasBaselineComparison
    public static void main(String[] args) throws Exception {
        Date date = new Date();
        int[] Ms =  new int[]{10000, 5000, 4000, 3000, 2000, 1000 };
        int[] Ns = new int[]{1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10, 5, 2, 1};
        double numAvg = 5.0;

        String dataset = args[0];
        System.out.println(dataset);

        double propnTrain = 0.50;

        DataLabel input = getLabeledData(dataset);


        double[][] totalRuntimes = new double[Ms.length + 1][Ns.length + 1];

        for (int i = 0; i < Ms.length; i++) {
            totalRuntimes[i+1][0] = Ms[i];
        }
        for (int j = 0; j < Ns.length; j++) {
            totalRuntimes[0][j+1] = Ns[j];
        }

        for (int i = 0; i < Ms.length; i++) {
            for (int j = 0; j < Ns.length; j++) {
                int M = Ms[i];
                int N = Ns[j];
                KMeans classifier = new KMeans(input.truncate(M, N), 5, propnTrain);

                for (int trial = 0; trial < numAvg; trial++) {
                    classifier.runKmeans();
                    totalRuntimes[i + 1][j + 1] += classifier.getTrainTime() + classifier.getAssignTime();
                }

                totalRuntimes[i + 1][j + 1] /= numAvg;
                System.out.print(classifier.M);
                System.out.print(" ");
                System.out.println(classifier.N);
                System.out.print(" ");
                System.out.print(totalRuntimes[i + 1][j + 1]);
                System.out.println(" ");
            }
        }

        double2dListToCSV(totalRuntimes, runtimeOutFile(date, dataset));

    }
}
