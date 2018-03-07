package experiments;

import runner.FFTRunner;
import runner.PAARunner;
import runner.PCARunner;
import optimizer.PCASkiingOptimizer;
import org.apache.commons.math3.linear.RealMatrix;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

/**
 * Created by meep_me on 9/1/16.
 */
public class BatchTechniqueComparison extends Experiment {
    public static String baseString = "contrib/src/main/java/macrobase/analysis/stats/optimizer/experiments/truncatedInputBaselineExperiments/";
    public static DateFormat day = new SimpleDateFormat("MM-dd");
    public static DateFormat minute = new SimpleDateFormat("HH_mm");

    private static String LBROutFile(String dataset, double qThresh, String tag, Date date){
        String output = String.format("%s_%s_q%.3f_%s",minute.format(date),dataset,qThresh, tag);
        return String.format(baseString + day.format(date) + "/KvLBR/%s.csv", output);
    }

    private static String timeOutFile(String dataset, double qThresh, String tag, Date date){
        String output = String.format("%s_%s_q%.3f_%s", minute.format(date), dataset,qThresh, tag);
        return String.format(baseString + day.format(date) + "/KvTime/%s.csv", output);
    }

    //java ${JAVA_OPTS} -cp "assembly/target/*:core/target/classes:frontend/target/classes:contrib/target/classes" macrobase.analysis.stats.experiments.SVDDropExperiments
    public static void main(String[] args) throws Exception{
        Date date = new Date();

        String dataset = args[0];
        double lbr = Double.parseDouble(args[1]);
        double qThresh = Double.parseDouble(args[2]);
        System.out.println(dataset);
        System.out.println(lbr);
        System.out.println(qThresh);

        Map<String,Map<Integer, Double>> fftResults;
        Map<String,Map<Integer, Double>> paaResults;
        //Map<String,Map<Integer, Double>> rpResults;
        Map<String,Map<Integer, Double>> pcaResults;
        Map<String,Map<Integer, Double>> pcaTroppResults;
        Map<String,Map<Integer, Double>> pcaFastResults;


        RealMatrix data = getData(dataset);

        PAARunner paaDrop = new PAARunner(qThresh, lbr);
        FFTRunner fftDrop = new FFTRunner(qThresh, lbr);
        //runner.RPRunner rpSkiingDROP = new runner.RPRunner(conf, qThresh, lbr);
        PCARunner pcaDrop = new PCARunner(qThresh, lbr, PCASkiingOptimizer.PCAAlgo.SVD);
        PCARunner pcaTroppDrop = new PCARunner(qThresh, lbr, PCASkiingOptimizer.PCAAlgo.TROPP);
        PCARunner pcaFastDrop = new PCARunner(qThresh, lbr, PCASkiingOptimizer.PCAAlgo.FAST);



        paaResults = paaDrop.genBasePlots(data);
        fftResults = fftDrop.genBasePlots(data);
        pcaResults = pcaDrop.genBasePlots(data);
        pcaTroppResults = pcaTroppDrop.genBasePlots(data);
        pcaFastResults = pcaFastDrop.genBasePlots(data);
        //rpResults = rpSkiingDROP.genBasePlots(data);

        mapDoubleToCSV(paaResults.get("LBR"), LBROutFile(dataset,qThresh,"PAA", date));
        mapDoubleToCSV(fftResults.get("LBR"), LBROutFile(dataset,qThresh, "FFT", date));
        //mapDoubleToCSV(rpResults.get("LBR"),  LBROutFile(dataset,qThresh, "RP", date));
        mapDoubleToCSV(pcaResults.get("LBR"), LBROutFile(dataset,qThresh, "PCASVD", date));
        mapDoubleToCSV(pcaTroppResults.get("LBR"), LBROutFile(dataset,qThresh, "PCATROPP", date));
        mapDoubleToCSV(pcaFastResults.get("LBR"), LBROutFile(dataset,qThresh, "PCAFAST", date));


        mapDoubleToCSV(paaResults.get("time"), timeOutFile(dataset,qThresh,"PAA", date));
        mapDoubleToCSV(fftResults.get("time"), timeOutFile(dataset,qThresh, "FFT", date));
        //mapDoubleToCSV(rpResults.get("time"),  timeOutFile(dataset,qThresh, "RP", date));
        mapDoubleToCSV(pcaResults.get("time"), timeOutFile(dataset,qThresh, "PCASVD", date));
        mapDoubleToCSV(pcaTroppResults.get("time"), timeOutFile(dataset,qThresh, "PCATROPP", date));
        mapDoubleToCSV(pcaFastResults.get("time"), timeOutFile(dataset,qThresh, "PCAFAST", date));
        

    }

}
