 package hw3;

import javax.management.RuntimeErrorException;

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
	 
	try{ if (aColumns != bColumns || aRows != bRows)
	 {
		 throw new IllegalArgumentException("A:Columns: " + aColumns + "did not math B:Columns " +bRows + ".");
		}
	}
	catch(IllegalArgumentException e){
	 System.out.println("Row " + aRows + " Col "+ aColumns );
	 System.out.println("Row " + bRows + " Col "+ bColumns );
	 throw new IllegalAccessError();
	}
	 
	 double returnVal = 0.0;
	 
	 for(int i =0; i<aRows;i++)
	 {
		 for(int j = 0; j<bColumns ; j++)
		 {
			
			 try {returnVal += A[i][j] *B[i][j];}
			 catch(NullPointerException e)
			 {System.out.print(A[i][j]);}
		 }
	 }
	 
	
	return returnVal;
 }
 
 
 //Calculate the result for two vectors
 public static double Dot(Double[] A, Double[] B)
 {
	 int aRows = A.length;
	 
	 int bRows = B.length;
	
	 
	 if (aRows != bRows)
	 {
		 throw new IllegalArgumentException("Rows doesn't match!");
	 }
	 
	 double returnVal = 0.0;
	 
	 for(int i =0; i<aRows;i++)
	 {
			 returnVal += A[i] *B[i];
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
 
