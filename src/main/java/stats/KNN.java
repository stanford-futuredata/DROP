package stats;

import com.google.common.base.Stopwatch;
import smile.math.distance.Metric;
import smile.neighbor.CoverTree;
import smile.neighbor.Neighbor;
import utils.CSVTools.*;


import org.apache.commons.math3.stat.StatUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Created by sahaana on 2/5/18.
 */
public class KNN {

    CoverTree<double[]> knn;
    double[][] trainData;
    double[][] testData;
    public int M; //number of data points
    public int N; //data dimension
    int numTrainPts; //number of datapts for train
    int numTestPts; //number of datapts for test
    Map<Integer, String> inputDataLabels;
    Map<Integer, String> outputDataLabels;
    Stopwatch trainTime;
    Stopwatch KNNTime;

    public KNN(DataLabel inputData, Metric distanceMeasure, double trainPropn) throws Exception {
        trainTime = Stopwatch.createUnstarted();
        KNNTime = Stopwatch.createUnstarted();

        DataLabel input = inputData;
        double[][] allData = input.matrix;
        inputDataLabels = input.labelMap;
        this.M = allData.length;
        this.N = allData[0].length;
        numTrainPts = (int) Math.round(trainPropn * M);
        numTestPts = M - numTrainPts;

        double[][][] splitData = splitRows(allData, numTrainPts, numTestPts, N);
        trainData = splitData[0];
        testData = splitData[1];

        trainTime.start();
        knn = new CoverTree<>(trainData,distanceMeasure);
        trainTime.stop();
    }

    public void runKNN(int k) throws Exception {
        runKNN(k, 1);
    }

    public void runKNN(int k, int numCycles) throws Exception {
        KNNTime.start();
        for (int cycles = 0; cycles < numCycles; cycles++) {
            outputDataLabels = new HashMap<>();
            if (k == 1) {
                for (int i = 0; i < numTestPts; i++) {
                    Neighbor<double[], double[]> nn = knn.nearest(testData[i]);
                    outputDataLabels.put(i, inputDataLabels.get(nn.index));
                }
            } else {
                for (int i = 0; i < numTestPts; i++) {
                    Neighbor<double[], double[]>[] nns = knn.knn(testData[i], k);
                    double[] nIndices = new double[k];
                    for (int n = 0; n < k; n++) {
                        nIndices[n] = nns[n].index;
                    }
                    outputDataLabels.put(i, inputDataLabels.get((int) StatUtils.mode(nIndices)[0]));
                }
            }
        }
        KNNTime.stop();
    }

    public double knnAccuracy() {
        double count = 0;
        double match = 0;
        for (int i = 0; i < numTestPts; i++) {
            count++;
            if (inputDataLabels.get(i + numTrainPts).equals(outputDataLabels.get(i))) {
                match++;
            }
        }
        return match/count;
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

    public double getKNNTime() {
        return KNNTime.elapsed(TimeUnit.MILLISECONDS);
    }
}
