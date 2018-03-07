package demo;

import experiments.Experiment;
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

public class DROPDemo extends Experiment {
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");

    private static String transformOutFile(String dataset, double lbr, double qThresh, PCASkiingOptimizer.PCAAlgo algo, PCASkiingOptimizer.work reuse, Date date, PCASkiingOptimizer.optimize opt){
        String output = String.format("%s_%s_%s_lbr%.2f_q%.2f_%s_%s",minute.format(date),dataset, algo, lbr, qThresh, reuse, opt);
        return String.format("data/transformed/"+day.format(date)+"/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception {
        Date date = new Date();

        //Input arguments: data, LBR, numtrials, and opt or not
        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);

        //Fixed arguments
        double qThresh = 1.96;
        PCASkiingOptimizer.optimize opt = PCASkiingOptimizer.optimize.OPTIMIZE;
        PCASkiingOptimizer.PCAAlgo algo = PCASkiingOptimizer.PCAAlgo.TROPP;
        PCASkiingOptimizer.work reuse = PCASkiingOptimizer.work.NOREUSE;
        Metric<double[]> distanceMeasure = new EuclideanDistance();
        double propnTrain = 0.5;

        //PlaceHolders
        RealMatrix data;
        String labels[];
        KNN knnClassifier;

        //load data and get labels as list
        CSVTools.DataLabel d = getLabeledData(dataset);
        data = new Array2DRowRealMatrix(d.matrix);
        labels = d.origLabels;

        PCARunner drop = new PCARunner(qThresh, lbr, algo, reuse, opt);
        drop.consume(data);
        double2dListToCSV(drop.getLabeledFinalTransform(labels), transformOutFile(dataset, lbr, qThresh, algo, reuse, date, opt));
        d = new CSVTools.DataLabel(drop.getFinalTransform(), labels, data.getRowDimension(), drop.finalK());
        knnClassifier = new KNN(d, distanceMeasure, propnTrain);
        knnClassifier.runKNN(1);

        System.out.print("Input: ");
        System.out.println(dataset);
        System.out.print("TLB: ");
        System.out.println(lbr);
        System.out.print("DROP Runtime: ");
        System.out.println(drop.totalTime());
        System.out.print("Returned Dimensionality: ");
        System.out.println(drop.finalK());
        System.out.print("KNN Runtime: ");
        System.out.println(knnClassifier.getKNNTime() + knnClassifier.getTrainTime());
        System.out.print("KNN Accuracy: ");
        System.out.println(knnClassifier.knnAccuracy());

        }
    }

