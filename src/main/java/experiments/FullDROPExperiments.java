package experiments;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.KNN;
import utils.CSVTools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by meep_me on 9/1/16.
 */
public class FullDROPExperiments extends Experiment {
    public static String baseString = "src/main/java/experiments/DROPExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");

    private static String allOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_%s_%s",minute.format(date),dataset, algo, lbr, qThresh, reuse, opt);
        return String.format(baseString + day.format(date) + "/%s.csv", output);
    }

    private static String transformOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt, int iter){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_%s_%s_%d",minute.format(date),dataset, algo, lbr, qThresh, reuse, opt, iter);
        return String.format("/lfs/1/sahaana/data/MIC_DROP/transformed/"+day.format(date)+"/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception {
        Date date = new Date();

        //Input arguments: data, LBR, numtrials, and opt or not
        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.valueOf(args[2]);
        int numTrials = Integer.parseInt(args[3]);
        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(opt);
        System.out.println(numTrials);

        //Fixed arguments:
        double qThresh = 1.96;
        PCASkiingOptimizer.PCAAlgo[] algos = {PCASkiingOptimizer.PCAAlgo.TROPP};//, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.work[] resuseStates = {PCASkiingOptimizer.work.NOREUSE};//, PCASkiingOptimizer.work.REUSE};
        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.75;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        double[] avgDROPRuntime;
        double[] avgK;
        double[] avgAccuracy;
        double[] avgKNNRuntime;
        double[] avgNt;
        double[] avgIters;
        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;


        for (PCASkiingOptimizer.PCAAlgo algo : algos) {
            for (PCASkiingOptimizer.work reuse : resuseStates) {
                avgDROPRuntime = new double[numTrials];
                avgK = new double[numTrials];
                avgAccuracy = new double[numTrials];
                avgKNNRuntime = new double[numTrials];
                avgNt = new double[numTrials];
                avgIters = new double[numTrials];

                for (int i = 0; i < numTrials; i++) {
                    PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
                    drop.consume(data);

                    //update DROP stuff
                    avgK[i] = drop.finalK();
                    avgDROPRuntime[i] = drop.totalTime();
                    avgNt[i] = drop.getNt();
                    avgIters[i] = drop.getNumIters();
                    double2dListToCSV(drop.getLabeledFinalTransform(labels), transformOutFile(dataset, lbr, qThresh, algo, reuse, date, opt, i));

                    d = new CSVTools.DataLabel(drop.getFinalTransform(), labels, data.getRowDimension(), drop.finalK());
                    knnClassifier = new KNN(d, distanceMeasure, propnTrain);
                    knnClassifier.runKNN(1);
                    avgKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                    avgAccuracy[i] = knnClassifier.knnAccuracy();
                }

                double[][] outputValues = new double[][]{avgDROPRuntime, avgK, avgNt, avgIters, avgAccuracy, avgKNNRuntime};
                double2dListToCSV(outputValues, allOutFile(dataset, lbr, qThresh, algo, reuse, date, opt));
            }
        }
    }
}
