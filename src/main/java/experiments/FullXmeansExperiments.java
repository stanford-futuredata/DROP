package experiments;

import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import runner.FFTRunner;
import runner.PAARunner;
import runner.PCARunner;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.Metric;
import stats.XMeans;
import utils.CSVTools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by meep_me on 9/1/16.
 */
public class FullXmeansExperiments extends Experiment {
    public static String baseString = "src/main/java/experiments/e2eXMeansExperiments/";
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

    private static String allDROPOutFile(String dataset, double lbr, double qThresh, Date date){
        String output = String.format("%s_%s_lbr%.2f_q%.2f",minute.format(date),dataset, lbr, qThresh);
        return String.format(baseString + day.format(date) + "/DROP/%s.csv", output);
    }

    private static String noDROutFile(String dataset, Date date){
        String output = String.format("%s_%s",minute.format(date),dataset);
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
        boolean knn = false;
        double qThresh = 1.96;
        double propnTrain = 0.75;
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.TROPP;//, PCASkiingOptimizer.PCAAlgo.FAST};
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.OPTIMIZE;
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.REUSE;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        double[] avgDROPRuntime = new double[numTrials];
        double[] avgDROPK = new double[numTrials];
        double[] avgDROPAccuracy = new double[numTrials];
        double[] avgDROPKNNRuntime = new double[numTrials];
        double[] avgPAARuntime = new double[numTrials];
        double[] avgPAAK = new double[numTrials];
        double[] avgPAAAccuracy = new double[numTrials];
        double[] avgPAAKNNRuntime = new double[numTrials];
        double[] avgFFTRuntime = new double[numTrials];
        double[] avgFFTK = new double[numTrials];
        double[] avgFFTAccuracy = new double[numTrials];
        double[] avgFFTKNNRuntime = new double[numTrials];
        double[] avgRuntime = new double[numTrials];
        double[] avgKNNRuntime = new double[numTrials];
        double[] avgK = new double[numTrials];
        double[] avgAccuracy = new double[numTrials];
        XMeans XMeansClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;

        for (int i = 0; i < numTrials; i++) {
            //no DR
            XMeansClassifier = new XMeans(d, 30, propnTrain);
            XMeansClassifier.runXmeans();
            avgKNNRuntime[i] = XMeansClassifier.getAssignTime() + XMeansClassifier.getTrainTime();
            avgAccuracy[i] = XMeansClassifier.getDistortion();
            avgK[i] = d.N;
            avgRuntime[i] = 0;

            PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt, PCASkiingOptimizer.sampling.SAMPLE, knn);
            PAARunner paa = new PAARunner(qThresh, lbr);
            FFTRunner fft = new FFTRunner(qThresh, lbr);

            drop.consume(data);
            //paa.consume(data);
            //fft.consume(data);

            //update DROP stuff
            avgDROPK[i] = drop.finalK();
            avgDROPRuntime[i] = drop.totalTime();

            d = new CSVTools.DataLabel(drop.getFinalTransform(), labels, data.getRowDimension(), drop.finalK());
            XMeansClassifier = new XMeans(d, 30, propnTrain);
            XMeansClassifier.runXmeans();
            avgDROPKNNRuntime[i] = XMeansClassifier.getAssignTime() + XMeansClassifier.getTrainTime();
            avgDROPAccuracy[i] = XMeansClassifier.getDistortion();

            /*
            //update PAA stuff
            avgPAAK[i] = paa.finalK();
            avgPAARuntime[i] = paa.totalTime();

            d = new CSVTools.DataLabel(paa.getFinalTransform(), labels, data.getRowDimension(), paa.finalK());
            XMeansClassifier = new XMeans(d, 30, propnTrain);
            XMeansClassifier.runXmeans();
            avgPAAKNNRuntime[i] = XMeansClassifier.getAssignTime() + XMeansClassifier.getTrainTime();
            avgPAAAccuracy[i] = XMeansClassifier.getDistortion();

            //update FFT stuff
            avgFFTK[i] = fft.finalK();
            avgFFTRuntime[i] = fft.totalTime();

            d = new CSVTools.DataLabel(fft.getFinalTransform(), labels, data.getRowDimension(), fft.finalK());
            XMeansClassifier = new XMeans(d, 30, propnTrain);
            XMeansClassifier.runXmeans();
            avgFFTKNNRuntime[i] = XMeansClassifier.getAssignTime() + XMeansClassifier.getTrainTime();
            avgFFTAccuracy[i] = XMeansClassifier.getDistortion();
            */
        }

        //double[][] outputFFTValues = new double[][]{avgFFTRuntime, avgFFTK, avgFFTAccuracy, avgFFTKNNRuntime, avgFFTAccuracy};
        //double[][] outputPAAValues = new double[][]{avgPAARuntime, avgPAAK, avgPAAAccuracy, avgPAAKNNRuntime, avgPAAAccuracy};
        double[][] outputDROPValues = new double[][]{avgDROPRuntime, avgDROPK, avgDROPAccuracy, avgDROPKNNRuntime, avgDROPAccuracy};
        double[][] outputValues = new double[][]{avgRuntime, avgK, avgAccuracy, avgKNNRuntime, avgAccuracy};
        //double2dListToCSV(outputPAAValues, allPAAOutFile(dataset, lbr, qThresh, date));
        //double2dListToCSV(outputFFTValues, allFFTOutFile(dataset, lbr, qThresh, date));
        double2dListToCSV(outputDROPValues, allDROPOutFile(dataset, lbr, qThresh, date));
        double2dListToCSV(outputValues, noDROutFile(dataset, date));
    }
}
