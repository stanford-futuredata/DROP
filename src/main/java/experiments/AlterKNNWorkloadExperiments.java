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
public class AlterKNNWorkloadExperiments extends Experiment {
    public static String baseString = "src/main/java/experiments/AlterWorkloadExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String allPAAOutFile(String dataset, double lbr, double qThresh, Date date, int numKnn){
        String output = String.format("%s_%s_cycle%d_lbr%.2f_q%.2f",minute.format(date),dataset, numKnn, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/PAA/%s.csv", output);
    }

    private static String allFFTOutFile(String dataset, double lbr, double qThresh, Date date, int numKnn){
        String output = String.format("%s_%s_cycle%d_lbr%.2f_q%.2f",minute.format(date),dataset, numKnn, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/FFT/%s.csv", output);
    }

    private static String allDROPOutFile(String dataset, double lbr, double qThresh, Date date, int numKnn){
        String output = String.format("%s_%s_cycle%d_lbr%.2f_q%.2f",minute.format(date),dataset, numKnn, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/DROP/%s.csv", output);
    }

    private static String allOutFile(String dataset, Date date, int numKnn){
        String output = String.format("%s_%s_cycle%d",minute.format(date),dataset, numKnn);
        return String.format(baseString + day.format(date) + "/NoDR/%s.csv", output);
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
        int[] knnCycles = new int[]{50};
        double qThresh = 1.96;
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.TROPP;//, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.OPTIMIZE;
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.REUSE;

        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.5;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        CSVTools.DataLabel dd;
        double[] avgDROPRuntime;
        double[] avgDROPK;
        double[] avgDROPAccuracy;
        double[] avgDROPKNNRuntime;
        double[] avgPAARuntime;
        double[] avgPAAK;
        double[] avgPAAAccuracy;
        double[] avgPAAKNNRuntime;
        double[] avgFFTRuntime;
        double[] avgFFTK;
        double[] avgFFTAccuracy;
        double[] avgFFTKNNRuntime;
        double[] avgRuntime;
        double[] avgK;
        double[] avgAccuracy;
        double[] avgKNNRuntime;

        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;

        for (int numKnnQuery  : knnCycles) {

            avgDROPRuntime = new double[numTrials];
            avgDROPK = new double[numTrials];
            avgDROPAccuracy = new double[numTrials];
            avgDROPKNNRuntime = new double[numTrials];
            avgPAARuntime = new double[numTrials];
            avgPAAK = new double[numTrials];
            avgPAAAccuracy = new double[numTrials];
            avgPAAKNNRuntime = new double[numTrials];
            avgFFTRuntime = new double[numTrials];
            avgFFTK = new double[numTrials];
            avgFFTAccuracy = new double[numTrials];
            avgFFTKNNRuntime = new double[numTrials];
            avgRuntime = new double[numTrials];
            avgK = new double[numTrials];
            avgAccuracy = new double[numTrials];
            avgKNNRuntime = new double[numTrials];

            for (int i = 0; i < numTrials; i++) {
                //update no DR
                avgK[i] = 0;
                avgRuntime[i] = 0;
                knnClassifier = new KNN(d, distanceMeasure, propnTrain);
                knnClassifier.runKNN(1, numKnnQuery);
                avgKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                avgAccuracy[i] = knnClassifier.knnAccuracy();

                PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
                PAARunner paa = new PAARunner(qThresh, lbr);
                FFTRunner fft = new FFTRunner(qThresh, lbr);

                drop.consume(data);
                paa.consume(data);
                fft.consume(data);

                //update DROP stuff
                avgDROPK[i] = drop.finalK();
                avgDROPRuntime[i] = drop.totalTime();

                dd = new CSVTools.DataLabel(drop.getFinalTransform(), labels, data.getRowDimension(), drop.finalK());
                knnClassifier = new KNN(dd, distanceMeasure, propnTrain);
                knnClassifier.runKNN(1,numKnnQuery);
                avgDROPKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                avgDROPAccuracy[i] = knnClassifier.knnAccuracy();


                //update PAA stuff
                avgPAAK[i] = paa.finalK();
                avgPAARuntime[i] = paa.totalTime();

                dd = new CSVTools.DataLabel(paa.getFinalTransform(), labels, data.getRowDimension(), paa.finalK());
                knnClassifier = new KNN(dd, distanceMeasure, propnTrain);
                knnClassifier.runKNN(1,numKnnQuery);
                avgPAAKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                avgPAAAccuracy[i] = knnClassifier.knnAccuracy();


                //update FFT stuff
                avgFFTK[i] = fft.finalK();
                avgFFTRuntime[i] = fft.totalTime();

                dd = new CSVTools.DataLabel(fft.getFinalTransform(), labels, data.getRowDimension(), fft.finalK());
                knnClassifier = new KNN(dd, distanceMeasure, propnTrain);
                knnClassifier.runKNN(1,numKnnQuery);
                avgFFTKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
                avgFFTAccuracy[i] = knnClassifier.knnAccuracy();

            }
            double[][] outputValues = new double[][]{avgRuntime, avgK, avgAccuracy, avgKNNRuntime};
            double[][] outputFFTValues = new double[][]{avgFFTRuntime, avgFFTK, avgFFTAccuracy, avgFFTKNNRuntime};
            double[][] outputPAAValues = new double[][]{avgPAARuntime, avgPAAK, avgPAAAccuracy, avgPAAKNNRuntime};
            double[][] outputDROPValues = new double[][]{avgDROPRuntime, avgDROPK, avgDROPAccuracy, avgDROPKNNRuntime};
            double2dListToCSV(outputPAAValues, allPAAOutFile(dataset, lbr, qThresh, date, numKnnQuery));
            double2dListToCSV(outputFFTValues, allFFTOutFile(dataset, lbr, qThresh, date, numKnnQuery));
            double2dListToCSV(outputDROPValues, allDROPOutFile(dataset, lbr, qThresh, date, numKnnQuery));
            double2dListToCSV(outputValues, allOutFile(dataset, date, numKnnQuery));
        }
    }
}
