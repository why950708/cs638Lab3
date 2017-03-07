 package hw3;
 public class Util{
 public static Double[][] matrixMultiply(Double[][] A, Double[][] B) {

        int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = B[0].length;

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }

        Double[][] C = new Double[aRows][bColumns];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                C[i][j] = 0.00000;
            }
        }

        for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                for (int k = 0; k < aColumns; k++) { // aColumn
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
	}
	
//Calculate the result for the kernal 
 public static double matrixDot(Double[][] A, Double[][] B)
 {
	 int aRows = A.length;
	 int aColumns = A[0].length;
	 int bRows = B.length;
	 int bColumns = B[0].length;
	 
	 if (aColumns != bColumns || aRows != bRows)
	 {
		 throw new IllegalArgumentException("A:Rows: " + aColumns + "did not math B:Columns " +bRows + ".");
	 }
	 
	 double returnVal = 0.0;
	 
	 for(int i =0; i<aRows;i++)
	 {
		 for(int j = 0; j<bRows ; j++)
		 {
			 returnVal += A[i][j] *B[i][j];
		 }
	 }
	 
	
	return returnVal;
 }
 
 //Transpose matrix
 public static double[][] transposeMatrix(double [][] m){
        double[][] rst = new double[m[0].length][m.length];
		
        for (int i = 0; i < m.length; i++)
        
		for (int j = 0; j < m[0].length; j++)
        
			rst[j][i] = m[i][j];
        
		return rst;
    }
 
 }
 
