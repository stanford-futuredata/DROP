package stats;

import no.uib.cipr.matrix.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;


/**
 * Created by sahaana on 1/9/18.
 */
public class TransformHistory {
    PCATropp svd;
    RealMatrix history;
    boolean historyExists;
    int size;
    //int prevK;
    boolean feasible;
    boolean svdExists;
    int N;

    public TransformHistory() {
        historyExists = false;
        size = 0;
        feasible = false;
        svdExists = false;
        N = 0;
    }


    //As soon as we're feasible, smoosh down history. This only happens after
    public void setFeasible(int k) {
        history = history.getSubMatrix(0,N-1,0,k-1);
        //        new Array2DRowRealMatrix(Matrices.getArray(Matrices.getSubMatrix(m, makeIntList(N), makeIntList(k)).copy()));
        size = k;
    }

    //Every time a new transformation is computed, append it to history
    //Run PCA over the history so that it's the same size as the current transform
    //if feasible, then smoosh down to prevk. If not, then let it grow.
    public Matrix smoosh(Matrix currTransform, int K) {
        N = currTransform.numRows();
        if (historyExists) {
            int currCols = history.getColumnDimension();
            int thisK = currTransform.numColumns();

            RealMatrix newHistory = new Array2DRowRealMatrix(N, currCols + thisK);
            newHistory.setSubMatrix(Matrices.getArray(currTransform), 0, 0);
            newHistory.setSubMatrix(history.getData(), 0, thisK);

            //computing svd/doing PCA manually cuz we need the U matrix
            svd = new PCATropp(newHistory); // H = USV'
            svd.computeTransformationForReuse(K);
            history = svd.transform(newHistory, K);  // H = HV
            size = history.getColumnDimension(); //TODO: should just be k, but still modding history around
            svdExists = true;
            return svd.getUMatrix(); // new basis = U
        }

        //trusting that addition of the columns won't cause poor conditioning right away
        historyExists = true;
        history = new Array2DRowRealMatrix(Matrices.getArray(currTransform));
        size = history.getColumnDimension();
        return currTransform;
    }

    public int getSize(){
        return size;}
}
