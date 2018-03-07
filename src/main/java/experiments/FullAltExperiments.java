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
public class FullAltExperiments extends Experiment {
    public static String baseString = "src/main/java/experiments/AltExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");


    private static String allPAAOutFile(String dataset, double lbr, double qThresh, Date date){
        String output = String.format("%s_%s_lbr%.2f_q%.2f",minute.format(date),dataset, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/PAA/%s.csv", output);
    }

    private static String allFFTOutFile(String dataset, double lbr, double qThresh, Date date){
        String output = String.format("%s_%s_lbr%.2f_q%.2f",minute.format(date),dataset, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/FFT/%s.csv", output);
    }

    private static String transformPAAOutFile(String dataset, double lbr, double qThresh, Date date, int iter){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_%d",minute.format(date),dataset, lbr, qThresh, iter);
        return String.format("/lfs/1/sahaana/data/MIC_DROP/transformed/PAA/"+day.format(date)+"/%s.csv", output);
    }

    private static String transformFFTOutFile(String dataset, double lbr, double qThresh, Date date, int iter){
        String output = String.format("%s_%s_lbr%.2f_q%.2f_%d",minute.format(date),dataset, lbr, qThresh, iter);
        return String.format("/lfs/1/sahaana/data/MIC_DROP/transformed/FFT/"+day.format(date)+"/%s.csv", output);
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
        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.75;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        double[] avgPAARuntime = new double[numTrials];
        double[] avgPAAK = new double[numTrials];
        double[] avgPAAAccuracy = new double[numTrials];
        double[] avgPAAKNNRuntime = new double[numTrials];
        double[] avgFFTRuntime = new double[numTrials];
        double[] avgFFTK = new double[numTrials];
        double[] avgFFTAccuracy = new double[numTrials];
        double[] avgFFTKNNRuntime = new double[numTrials];
        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;

        for (int i = 0; i < numTrials; i++) {
            PAARunner paa = new PAARunner(qThresh, lbr);
            FFTRunner fft = new FFTRunner(qThresh, lbr);

            paa.consume(data);
            fft.consume(data);

            //update PAA stuff
            avgPAAK[i] = paa.finalK();
            avgPAARuntime[i] = paa.totalTime();
            double2dListToCSV(paa.getLabeledFinalTransform(labels), transformPAAOutFile(dataset, lbr, qThresh, date, i));

            d = new CSVTools.DataLabel(paa.getFinalTransform(), labels, data.getRowDimension(), paa.finalK());
            knnClassifier = new KNN(d, distanceMeasure, propnTrain);
            knnClassifier.runKNN(1);
            avgPAAKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
            avgPAAAccuracy[i] = knnClassifier.knnAccuracy();


            //update FFT stuff
            avgFFTK[i] = fft.finalK();
            avgFFTRuntime[i] = fft.totalTime();
            double2dListToCSV(fft.getLabeledFinalTransform(labels), transformFFTOutFile(dataset, lbr, qThresh, date, i));

            d = new CSVTools.DataLabel(fft.getFinalTransform(), labels, data.getRowDimension(), fft.finalK());
            knnClassifier = new KNN(d, distanceMeasure, propnTrain);
            knnClassifier.runKNN(1);
            avgFFTKNNRuntime[i] = knnClassifier.getKNNTime() + knnClassifier.getTrainTime();
            avgFFTAccuracy[i] = knnClassifier.knnAccuracy();
        }

        double[][] outputFFTValues = new double[][]{avgFFTRuntime, avgFFTK, avgFFTAccuracy, avgFFTKNNRuntime};
        double[][] outputPAAValues = new double[][]{avgPAARuntime, avgPAAK, avgPAAAccuracy, avgPAAKNNRuntime};
        double2dListToCSV(outputPAAValues, allPAAOutFile(dataset, lbr, qThresh, date));
        double2dListToCSV(outputFFTValues, allFFTOutFile(dataset, lbr, qThresh, date));
    }
}
