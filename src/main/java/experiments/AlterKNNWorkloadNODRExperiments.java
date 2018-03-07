package experiments;

import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import runner.FFTRunner;
import runner.PAARunner;
import runner.PCARunner;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.KNN;
import utils.CSVTools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by meep_me on 9/1/16.
 */
public class AlterKNNWorkloadNODRExperiments extends Experiment {
    public static String baseString = "src/main/java/experiments/AlterWorkloadExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String allOutFile(String dataset, Date date, int numKnn){
        String output = String.format("%s_%s_cycle%d",minute.format(date),dataset, numKnn);
        return String.format(baseString + day.format(date) + "/NoDR/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception {
        Date date = new Date();

        //Input arguments: data, LBR, numtrials, and opt or not
        String dataset = args[0];
        int numTrials = Integer.parseInt(args[1]);
        System.out.println(dataset);
        System.out.println(numTrials);

        //Fixed arguments:
        int[] knnCycles = new int[]{1, 3, 9};

        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.75;

        //PlaceHolders
        double[] avgRuntime;
        double[] avgK;
        double[] avgAccuracy;
        double[] avgKNNRuntime;

        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);

        for (int numKnnQuery  : knnCycles) {

            avgRuntime = new double[numTrials];
            avgK = new double[numTrials];
            avgAccuracy = new double[numTrials];
            avgKNNRuntime = new double[numTrials];

            for (int i = 0; i < numTrials; i++) {
                //update FFT stuff
                avgK[i] = d.N;
                avgRuntime[i] = 0;

                knnClassifier = new KNN(d, distanceMeasure, propnTrain);
                knnClassifier.runKNN(1,numKnnQuery);
                avgKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                avgAccuracy[i] = knnClassifier.knnAccuracy();

            }
            double[][] outputValues = new double[][]{avgRuntime, avgK, avgAccuracy, avgKNNRuntime};
            double2dListToCSV(outputValues, allOutFile(dataset, date, numKnnQuery));
        }
    }
}
