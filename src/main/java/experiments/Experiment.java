package experiments;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import utils.CSVTools;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;

import utils.CSVTools.*;


/**
 * Created by meep_me on 9/1/16.
 */
public abstract class Experiment {

    public static void mapIntToCSV(Map<Integer, Integer> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Integer, Integer> entry: dataMap.entrySet()) {
                writer.append(Integer.toString(entry.getKey()))
                        .append(',')
                        .append(Integer.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapDoubleToCSV(Map<Integer, Double> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Integer, Double> entry: dataMap.entrySet()) {
                writer.append(Integer.toString(entry.getKey()))
                        .append(',')
                        .append(Double.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapLongDoubleToCSV(Map<Long, Double> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Long, Double> entry: dataMap.entrySet()) {
                writer.append(Long.toString(entry.getKey()))
                        .append(',')
                        .append(Double.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapLongIntToCSV(Map<Long, Integer> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Long, Integer> entry: dataMap.entrySet()) {
                writer.append(Long.toString(entry.getKey()))
                        .append(',')
                        .append(Integer.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void doubleListToCSV(double[] vals, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (double val: vals) {
                writer.append(Double.toString(val))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void intListToCSV(int[] vals, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (int val: vals) {
                writer.append(Integer.toString(val))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void double2dListToCSV(double[][] vals, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol = System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (double[] val: vals) {
                for (int i = 0; i < val.length; i++) {
                    writer.append(Double.toString(val[i]));
                    if (i != val.length - 1) {
                        writer.append(',');
                    }
                }
                writer.append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }


    public static void mapDoubleLongToCSV(Map<Double, Long> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Double, Long> entry: dataMap.entrySet()) {
                writer.append(Double.toString(entry.getKey()))
                        .append(',')
                        .append(Long.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapDoubleIntToCSV(Map<Double, Integer> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Double, Integer> entry: dataMap.entrySet()) {
                writer.append(Double.toString(entry.getKey()))
                        .append(',')
                        .append(Integer.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapIntLongToCSV(Map<Integer, Long> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Integer, Long> entry: dataMap.entrySet()) {
                writer.append(Integer.toString(entry.getKey()))
                        .append(',')
                        .append(Long.toString(entry.getValue()))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapDouble2ToCSV(Map<Integer, double[]> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Integer, double[]> entry: dataMap.entrySet()) {
                writer.append(Integer.toString(entry.getKey()))
                        .append(',')
                        .append(Double.toString(entry.getValue()[0]))
                        .append(',')
                        .append(Double.toString(entry.getValue()[1]))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void mapDouble3ToCSV(Map<Integer, double[]> dataMap, String file){
        File f = new File(file);
        f.getParentFile().mkdirs();
        String eol =  System.getProperty("line.separator");
        try (Writer writer = new FileWriter(f)) {
            for (Map.Entry<Integer, double[]> entry: dataMap.entrySet()) {
                writer.append(Integer.toString(entry.getKey()))
                        .append(',')
                        .append(Double.toString(entry.getValue()[0]))
                        .append(',')
                        .append(Double.toString(entry.getValue()[1]))
                        .append(',')
                        .append(Double.toString(entry.getValue()[2]))
                        .append(eol);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }


    public static Map<Integer, Integer> scaleIntMap(Map<Integer, Integer> toScale, Map<Integer,Double> counts){
        Map<Integer, Integer> scaled = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry: toScale.entrySet()) {
            int key = entry.getKey();
            int kval = entry.getValue();
            Double newK = new Double(kval)/counts.get(key);
            scaled.put(key, newK.intValue());
        }
        return scaled;
    }

    public static Map<Integer, Double> scaleDoubleMap(Map<Integer, Double> toScale, Map<Integer,Double> counts){
        Map<Integer, Double> scaled = new HashMap<>();
        for (Map.Entry<Integer, Double> entry: toScale.entrySet()) {
            int key = entry.getKey();
            double kval = entry.getValue();
            Double newK = new Double(kval)/counts.get(key);
            scaled.put(key, newK);
        }
        return scaled;
    }

    public static Map<Integer, double[]> scaleDouble2Map(Map<Integer, double[]> toScale, Map<Integer,Double> counts){
        Map<Integer, double[]> scaled = new HashMap<>();
        for (Map.Entry<Integer, double[]> entry: toScale.entrySet()) {
            int key = entry.getKey();
            double real = entry.getValue()[0];
            double guess = entry.getValue()[1];
            scaled.put(key, new double[] {real/counts.get(key), guess/counts.get(key)});
        }
        return scaled;
    }

    public static Map<Integer, double[]> scaleDouble3Map(Map<Integer, double[]> toScale, Map<Integer,Double> counts){
        Map<Integer, double[]> scaled = new HashMap<>();
        for (Map.Entry<Integer, double[]> entry: toScale.entrySet()) {
            int key = entry.getKey();
            double e0 = entry.getValue()[0];
            double e1 = entry.getValue()[1];
            double e2 = entry.getValue()[2];
            scaled.put(key, new double[] {e0/counts.get(key), e1/counts.get(key), e2/counts.get(key)});
        }
        return scaled;
    }

    public static Map<Double, Integer> scaleIntMapWCount(Map<Double, Integer> toScale, int count){
        Map<Double, Integer> scaled = new HashMap<>();
        for (Map.Entry<Double, Integer> entry: toScale.entrySet()) {
            double key = entry.getKey();
            int val = entry.getValue();
            scaled.put(key, val/count);
        }
        return scaled;
    }

    public static Map<Double, Double> scaleDoubleMapWCount(Map<Double, Double> toScale, int count){
        Map<Double, Double> scaled = new HashMap<>();
        for (Map.Entry<Double, Double> entry: toScale.entrySet()) {
            double key = entry.getKey();
            double val = entry.getValue();
            scaled.put(key, val/count);
        }
        return scaled;
    }

    public static Map<Double, Long> scaleLongMapWCount(Map<Double, Long> toScale, int count){
        Map<Double, Long> scaled = new HashMap<>();
        for (Map.Entry<Double, Long> entry: toScale.entrySet()) {
            double key = entry.getKey();
            long val = entry.getValue();
            scaled.put(key, val/count);
        }
        return scaled;
    }

    public static RealMatrix getData(String dataset) throws Exception{
        CSVTools ingester = new CSVTools(String.format("data/labeled/%s.csv", dataset));
        DataLabel input = ingester.getLabeledData(0);
        return new Array2DRowRealMatrix(input.matrix);
    }

    public static DataLabel getLabeledData(String dataset) throws Exception{
        CSVTools ingester = new CSVTools(String.format("data/labeled/%s.csv", dataset));
        return ingester.getLabeledData(0);
    }

}
