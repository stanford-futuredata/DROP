package stats;

import com.google.common.base.Stopwatch;
import smile.neighbor.KDTree;
import utils.CSVTools.DataLabel;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Created by sahaana on 2/5/18.
 */
public class DBScan {
    smile.clustering.DBScan<double[]>  dbScan;
    double[][] allData;
    double[][] trainData;
    double[][] testData;
    public int M; //number of data points
    public int N; //data dimension
    Map<Integer, String> inputDataLabels;
    Map<Integer, String> outputDataLabels;
    Stopwatch trainTime;
    Stopwatch assignTime;
    int numTrainPts;
    int numTestPts;


    public DBScan(DataLabel inputData, double radius, double trainPropn) throws Exception {
        trainTime = Stopwatch.createUnstarted();
        assignTime = Stopwatch.createUnstarted();

        DataLabel input = inputData;
        allData = input.matrix;
        inputDataLabels = input.labelMap;
        this.M = allData.length;
        this.N = allData[0].length;
        numTrainPts = (int) Math.round(trainPropn * M);
        numTestPts = M - numTrainPts;

        double[][][] splitData = splitRows(allData, numTrainPts, numTestPts, N);
        trainData = splitData[0];
        testData = splitData[1];

        trainTime.start();
        dbScan = new smile.clustering.DBScan<>(trainData, new KDTree<>(trainData, trainData), 300, radius);
        trainTime.stop();
    }

    public void runDBSCAN() throws Exception {
        outputDataLabels = new HashMap<>();

        assignTime.start();
        for (int i = 0; i < numTestPts; i++) {
            outputDataLabels.put(i + numTrainPts, Integer.toString(dbScan.predict(testData[i])));
        }
        assignTime.stop();
    }

    public double[][][] splitRows(double[][] input, int numFirstSplit, int numSecondSplit, int N) {
        double[][] split1 = new double[numFirstSplit][N];
        double[][] split2 = new double[numSecondSplit][N];

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < numFirstSplit + numSecondSplit; i++) {
                if (i < numFirstSplit) {
                    split1[i][j] = input[i][j];
                } else {
                    split2[i - numFirstSplit][j] = input[i][j];
                }
            }
        }
        return new double[][][]{split1, split2};
    }

    public double getTrainTime() {
        return trainTime.elapsed(TimeUnit.MILLISECONDS);
    }

    public double getAssignTime() {
        return assignTime.elapsed(TimeUnit.MILLISECONDS);
    }
}
