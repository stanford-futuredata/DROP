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
public class SVDKnnBaselineExperiment extends Experiment {
    public static String baseString = "src/main/java/experiments/e2eKNNExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String allSVDOutFile(String dataset, double lbr, double qThresh, Date date){
        String output = String.format("%s_%s_lbr%.2f_q%.2f",minute.format(date),dataset, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/SVD/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception {
        Date date = new Date();

        //Input arguments: data, LBR, numtrials, and opt or not
        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        int numTrials = Integer.parseInt(args[2]);
        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(numTrials);

        //Fixed arguments:
        double qThresh = 1.96;
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.SVD;//, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.NOOPTIMIZE;
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.NOREUSE;
        PCASkiingOptimizer.sampling sample = PCASkiingOptimizer.sampling.NOSAMPLE;

        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.75;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        double[] avgSVDRuntime = new double[numTrials];
        double[] avgSVDK = new double[numTrials];
        double[] avgSVDAccuracy = new double[numTrials];
        double[] avgSVDKNNRuntime = new double[numTrials];
        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;

        for (int i = 0; i < numTrials; i++) {
            PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt, sample);
            drop.consume(data);

            //update DROP stuff
            avgSVDK[i] = drop.finalK();
            avgSVDRuntime[i] = drop.totalTime();

            d = new CSVTools.DataLabel(drop.getFinalTransform(), labels, data.getRowDimension(), drop.finalK());
            knnClassifier = new KNN(d, distanceMeasure, propnTrain);
            knnClassifier.runKNN(1);
            avgSVDKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
            avgSVDAccuracy[i] = knnClassifier.knnAccuracy();

        }

        double[][] outputSVDValues = new double[][]{avgSVDRuntime, avgSVDK, avgSVDAccuracy, avgSVDKNNRuntime};
        double2dListToCSV(outputSVDValues, allSVDOutFile(dataset, lbr, qThresh, date));
    }
}
