package utils;

import optimizer.FFTSkiingOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import stats.PCA;
import stats.PCASVD;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CSVTools {

    private String filename;
    private String splitBy;

    // Class to return both a matrix of data, and corresponding index in data -> label mapping
    public final static class DataLabel {
        public double[][] matrix;
        public String[] origLabels;
        public Map<Integer, String> labelMap;
        public int M;
        public int N;

        public double[][] PCAMatrix;
        public double[][] FFTMatrix;
        public boolean transformedPCA;
        public boolean transformedFFT;


        public DataLabel(double[][] mat, Map labels, int M, int N) {
            this.M = M;
            this.N = N;
            this.matrix = mat;
            this.labelMap = labels;
            //dumhack
            origLabels = new String[labelMap.size()];
            for (int i = 0; i < labelMap.size(); i++) {
                origLabels[i] = labelMap.get(i);
            }
        }

        public DataLabel(double[][] mat, String[] labels, int M, int N) {
            this.M = M;
            this.N = N;
            this.matrix = mat;
            this.origLabels = labels;
            //moardumhack
            this.labelMap = new HashMap<>();
            for (int i = 0; i < labels.length; i++) {
                labelMap.put(i, labels[i]);
            }
        }

        public DataLabel truncate(int M, int N) {
            double[][] truncMat = new double[M][N];
            Map<Integer, String> truncLab = new HashMap<>();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    truncMat[i][j] = matrix[i][j];
                }
                truncLab.put(i, labelMap.get(i));
            }
            return new DataLabel(truncMat, truncLab, M, N);
        }

        public DataLabel truncateFFT(int N) {

            if (!transformedFFT) {
                FFTSkiingOptimizer fft = new FFTSkiingOptimizer(1.96);
                fft.extractData(new Array2DRowRealMatrix(matrix));
                fft.fit(N);
                FFTMatrix = fft.transform(N).getData();
            }

            double[][] truncMat = new double[M][N];
            Map<Integer, String> truncLab = new HashMap<>();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    truncMat[i][j] = FFTMatrix[i][j];
                }
                truncLab.put(i, labelMap.get(i));
            }
            transformedFFT = true;
            return new DataLabel(truncMat, truncLab, M, N);
        }

        public DataLabel truncatePCA(int N) {
            RealMatrix dat = new Array2DRowRealMatrix(matrix);
            PCA pca = new PCASVD(dat);

            if (!transformedPCA) {
                PCAMatrix = pca.transform(dat, N).getData();
            }

            double[][] truncMat = new double[M][N];
            Map<Integer, String> truncLab = new HashMap<>();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    truncMat[i][j] = PCAMatrix[i][j];
                }
                truncLab.put(i, labelMap.get(i));
            }
            transformedPCA = true;
            return new DataLabel(truncMat, truncLab, M, N);
        }
    }

    public CSVTools(String filename){
        this.filename = filename;
        this.splitBy = ",";
    }

    public RealMatrix getData() throws Exception {
        int M;
        int N;
        String line;
        RealVector record;
        String[] readLine;
        RealMatrix output;
        List<RealVector> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));

        int n = 0;
        while ((line = br.readLine()) != null){
            int i = 0;
            readLine = line.split(splitBy);
            record = new ArrayRealVector(readLine.length);
            for (String entry: readLine){
                record.setEntry(i++, Double.parseDouble(entry));
            }
            data.add(n++, record);
        }
        n = 0;
        M = data.size();
        N = data.get(0).getDimension();
        output = new Array2DRowRealMatrix(M, N);
        for (RealVector v : data) {
            output.setRowVector(n++,v);
        }
        return output;
    }

    public DataLabel getLabeledData(int labelCol) throws Exception {
        int M;
        int N;

        Map<Integer, String> labels = new HashMap<>();

        String line;
        RealVector record;
        String[] readLine;
        RealMatrix output;
        List<RealVector> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));

        int n = 0;
        while ((line = br.readLine()) != null) {
            int i = 0;
            readLine = line.split(splitBy);
            record = new ArrayRealVector(readLine.length - 1);
            for (int j = 0; j < readLine.length; j++) {
                if (j == labelCol) {
                    labels.put(n, readLine[j]);
                } else {
                    record.setEntry(i++, Double.parseDouble(readLine[j]));
                }
            }
            data.add(n++, record);
        }
        n = 0;
        M = data.size();
        N = data.get(0).getDimension();
        output = new Array2DRowRealMatrix(M, N);
        for (RealVector v : data) {
            output.setRowVector(n++, v);
        }
        return new DataLabel(output.getData(), labels, M, N);
    }

}


