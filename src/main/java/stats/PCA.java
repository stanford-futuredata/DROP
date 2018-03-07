package stats;

import no.uib.cipr.matrix.Matrix;
import org.apache.commons.math3.linear.RealMatrix;

public interface PCA {

    int getN();

    int getM();

    //always stores the largest transformation computed thus far with this PCA object, which is specific to input data (the subset)
    Matrix getTransformation();

    RealMatrix transform(RealMatrix inputData, int K);

    RealMatrix transform(RealMatrix inputData, int K, TransformHistory history);

}
