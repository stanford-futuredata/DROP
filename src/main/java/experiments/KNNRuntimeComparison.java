package experiments;

import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.KNN;
import utils.CSVTools.DataLabel;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by sahaana on 12/7/17.
 */
public class KNNRuntimeComparison extends Experiment {
    public static String baseString = "src/main/java/experiments/KNNCostModel/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String FFTruntimeOutFile(Date date, String dataset){
        String output = String.format("%s_%s",minute.format(date),dataset);
        return String.format(baseString + day.format(date) + "/FFT/%s.csv", output);
    }

    private static String PCAruntimeOutFile(Date date, String dataset){
        String output = String.format("%s_%s",minute.format(date),dataset);
        return String.format(baseString + day.format(date) + "/PCA/%s.csv", output);
    }

    //java -Xms8g ${JAVA_OPTS} -cp 'target/classes/' experiments.OjasBaselineComparison
    public static void main(String[] args) throws Exception {
        Date date = new Date();
        int[] Ms = new int[]{10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000};
        int[] Ns = new int[]{1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10, 5, 2, 1};
        double numAvg = 5.0;

        String dataset = args[0];
        System.out.println(dataset);

        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.75;

        DataLabel origInput = getLabeledData(dataset);

        double[][] FFTtotalRuntimes = new double[Ms.length + 1][Ns.length + 1];
        double[][] PCAtotalRuntimes = new double[Ms.length + 1][Ns.length + 1];

        for (int j = 0; j < Ns.length; j++) {
            FFTtotalRuntimes[0][j+1] = Ns[j];
            PCAtotalRuntimes[0][j+1] = Ns[j];
        }

        for (int i = 0; i < Ms.length; i++) {
            FFTtotalRuntimes[i+1][0] = Ms[i];
            PCAtotalRuntimes[i+1][0] = Ms[i];
        }

        for (int i = 0; i < Ms.length; i++) {
            DataLabel input = origInput.truncate(Ms[i], origInput.N);
            for (int j = 0; j < Ns.length; j++) {
                System.out.println(j);
                int N = Ns[j];
                KNN FFTclassifier = new KNN(input.truncateFFT(N), distanceMeasure, propnTrain);
                KNN PCAclassifier = new KNN(input.truncatePCA(N), distanceMeasure, propnTrain);

                for (int trial = 0; trial < numAvg; trial++) {
                    FFTclassifier.runKNN(1);
                    FFTtotalRuntimes[1][j + 1] += FFTclassifier.getTrainTime() + FFTclassifier.getKNNTime();

                    PCAclassifier.runKNN(1);
                    PCAtotalRuntimes[1][j + 1] += PCAclassifier.getTrainTime() + FFTclassifier.getKNNTime();
                }

                FFTtotalRuntimes[1][j + 1] /= numAvg;
                PCAtotalRuntimes[1][j + 1] /= numAvg;
            }
        }

        double2dListToCSV(FFTtotalRuntimes, FFTruntimeOutFile(date, dataset));
        double2dListToCSV(PCAtotalRuntimes, PCAruntimeOutFile(date, dataset));
    }
}
