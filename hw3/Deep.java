package hw3;

import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import javafx.util.Pair;
import hw3.Lab3;

/**
 * Do not modify.
 * 
 * This is the class with the main function
 */

public class Deep {
	/**
	 * Runs the tests for HW4
	 */

	Double[][][] c1weights;
	Double[][][] c2weights;
	Double[][][][] fl1weights;
	Double[][] fl2weights;
	Double[][][] p1toc2weights;

	Double[][][] TrainSet;
	Double[][][] TuneSet;
	Double[][][] TestSet;

	Double[] TrainLabel;
	Double[] TuneLabel;
	Double[] TestLabel;

	Double learningRate;

	Double[] biasC1;
	Double[] biasMP1;
	Double[] biasC2;
	Double[] biasMP2;
	Double[] biasFL;
	Double[] biasFL2;

	NNImpl nn;

	int numC1 = 20;
	int numMP1 = 20;
	int numC2 = 20;
	int numMP2 = 20;
	int numFL = 300;
	int numFL2 = 6;
	int imageSize = 32;
	int stride = 1;

	int C1KernalSize = 5;
	int C2KernalSize = 5;
	int C3KernalSize = 3;
	int MP1KernalSize = 2;
	int MP2KernalSize = 2;

	public Deep(Vector<Vector<Double>> trainfeatureVectors, Vector<Vector<Double>> tunefeatureVectors,
			Vector<Vector<Double>> testfeatureVectors) {

		// Reading the training set
		// trainingSet = getData(trainfeatureVectors);

		// Randomize the input
		Lab3.permute(trainfeatureVectors);
		Lab3.permute(tunefeatureVectors);
		Lab3.permute(testfeatureVectors);

		// Creating weights arrays
		c1weights = new Double[numC1][C1KernalSize][C1KernalSize];
		p1toc2weights = new Double[numMP1][MP1KernalSize][MP1KernalSize];
		c2weights = new Double[numC2][C2KernalSize][C2KernalSize];
		fl1weights = new Double[numFL][numMP2][MP2KernalSize][MP2KernalSize];
		fl2weights = new Double[numFL][numFL2];
		biasC1 = new Double[numC1];
		biasC2 = new Double[numC2];
		biasMP1 = new Double[numMP1];
		biasMP2 = new Double[numMP2];
		biasFL = new Double[numFL];
		biasFL2 = new Double[numFL2];

		// Creating images
		TrainSet = new Double[trainfeatureVectors.size()][imageSize][imageSize];
		TuneSet = new Double[tunefeatureVectors.size()][imageSize][imageSize];
		TestSet = new Double[testfeatureVectors.size()][imageSize][imageSize];

		// Creating label arrays
		Double[] TrainLabel = new Double[TrainSet.length];
		Double[] TuneLabel = new Double[TuneSet.length];
		Double[] TestLabel = new Double[TestSet.length];

		// Training Set
		for (int z = 0; z < trainfeatureVectors.size(); z++) {
			for (int i = 0; i < imageSize; i++)
				for (int j = 0; j < imageSize; j++) {
					// HOngyi Wang
					// Only getting the grayscale for the first part
					TrainSet[z][i][j] = Lab3.get2DfeatureValue(trainfeatureVectors.get(z), i, j, 0);
				}
			TrainLabel[z] = trainfeatureVectors.get(z).lastElement();
		}

		// Tune Set
		for (int z = 0; z < tunefeatureVectors.size(); z++) {
			for (int i = 0; i < imageSize; i++)
				for (int j = 0; j < imageSize; j++) {
					// HOngyi Wang
					// Only getting the grayscale for the first part
					TuneSet[z][i][j] = Lab3.get2DfeatureValue(tunefeatureVectors.get(z), i, j, 0);
				}
			TuneLabel[z] = tunefeatureVectors.get(z).lastElement();
		}

		// Test Set
		for (int z = 0; z < testfeatureVectors.size(); z++) {
			for (int i = 0; i < imageSize; i++)
				for (int j = 0; j < imageSize; j++) {
					// HOngyi Wang
					// Only getting the grayscale for the first part
					TestSet[z][i][j] = Lab3.get2DfeatureValue(testfeatureVectors.get(z), i, j, 0);
				}
			TestLabel[z] = testfeatureVectors.get(z).lastElement();
		}

		// Initilize Weights
		Random r = new Random();
		// C1Weights
		for (int i = 0; i < c1weights.length; i++) {
			for (int j = 0; j < c1weights[0].length; j++) {
				for (int k = 0; k < c1weights[0][0].length; k++) {
					c1weights[i][j][k] = r.nextDouble() * 0.01;
				}
			}
		}
		// p1toc2weights
		for (int i = 0; i < p1toc2weights.length; i++) {
			for (int j = 0; j < p1toc2weights[0].length; j++) {
				for (int k = 0; k < p1toc2weights[0][0].length; k++) {
					p1toc2weights[i][j][k] = r.nextDouble() * 0.01;
				}
			}
		}

		// c2weights
		for (int i = 0; i < c2weights.length; i++) {
			for (int j = 0; j < c2weights[0].length; j++) {
				for (int k = 0; k < c2weights[0][0].length; k++) {
					c2weights[i][j][k] = r.nextDouble() * 0.01;
				}
			}
		}

		// fl1weights
		for (int i = 0; i < fl1weights.length; i++) {
			for (int j = 0; j < fl1weights[0].length; j++) {
				for (int k = 0; k < fl1weights[0][0].length; k++) {
					for (int l = 0; l < fl1weights[0][0][0].length; l++) {
						fl1weights[i][j][k][l] = r.nextDouble() * 0.01;
					}
				}
			}
		}

		// fl2weights
		for (int i = 0; i < fl2weights.length; i++) {
			for (int j = 0; j < fl2weights[0].length; j++) {
				fl2weights[i][j] = r.nextDouble() * 0.01;
				// System.out.print(fl2weights[i][j]);
			}
		}

		for (int i = 0; i < biasC1.length; i++)
			biasC1[i] = r.nextDouble() * 0.01;
		for (int i = 0; i < biasC2.length; i++)
			biasC2[i] = r.nextDouble() * 0.01;
		// for (int i=0; i < biasMP1.length; i++) biasMP1[i] =
		// r.nextDouble()*0.01;
		// for (int i=0; i < biasMP2.length; i++) biasMP2[i] =
		// r.nextDouble()*0.01;
		for (int i = 0; i < biasFL.length; i++)
			biasFL[i] = r.nextDouble() * 0.01;
		for (int i = 0; i < biasFL2.length; i++)
			biasFL2[i] = r.nextDouble() * 0.01;

		forward(TrainSet[0]);

	}

	// Calculates the output for the network
	public Double[] forward(Double[][] input) {
		// Final result;
		Double[] rst = new Double[this.numFL2];

		// input to convolution layer output
		int size = (imageSize - C1KernalSize) / stride + 1;

		Double[][][] bufferC1 = new Double[numC1][size][size];

		for (int z = 0; z < c1weights.length; z++) {
			for (int i = 0; i < size; i++)
				for (int j = 0; j < size; j++) {
					// convolve
					// Get the sub array
					Double[][] a = new Double[c1weights[0].length][];
					for (int k = 0; k < a.length; k++) {
						a[k] = Arrays.copyOfRange(input[i], j, j + a.length);
					}
					double sum = Util.matrixDot(a, c1weights[z]);
					// Bias
					bufferC1[z][i][j] = sum - biasC1[z];

					// ReLu
					bufferC1[z][i][j] = sum < 0 ? 0.0 : sum;

				}
		}

		// c1 to mp1
		stride = 2;
		size = (size - MP1KernalSize) / stride + 1;
		// Hongyi Wang print size for debugging;
		// System.out.print(size);

		double max = Double.NEGATIVE_INFINITY;
		Double bufferMP1[][][] = new Double[numMP1][size][size];

		for (int z = 0; z < numMP1; z++) {
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					max = Double.NEGATIVE_INFINITY;
					for (int k = 0; k < MP1KernalSize; k++) {
						for (int l = 0; l < MP1KernalSize; l++) {
							if (bufferC1[z][i * stride + k][j * stride + l] > max) {
								max = bufferC1[z][i * stride + k][j * stride + l];
							}
						}
						bufferMP1[z][i][j] = max;
						// Hongyi Wang print max for debugging;
						// System.out.print(max);
					}

				}
			}
		}

		// mp1 to c2
		stride = 1;
		size = (size - C2KernalSize) / stride + 1;
		double sum = 0;
		Double[][][] bufferC2 = new Double[numC2][size][size];
		for (int z = 0; z < numC2; z++) {

			for (int i = 0; i < size; i++)
				for (int j = 0; j < size; j++) {
					for (int y = 0; y < numC1; y++) {
						// convolve
						// Get the sub array
						sum = 0;
						Double[][] a = new Double[C2KernalSize][];
						for (int k = 0; k < C1KernalSize; k++) {
							a[k] = Arrays.copyOfRange(bufferMP1[y][i], j, j + C2KernalSize);
						}
						// Convolution
						sum += Util.matrixDot(a, c2weights[z]);

					}
					bufferC2[z][i][j] = sum;
					// Bias
					bufferC2[z][i][j] -= biasC2[z];

					// ReLu
					if (bufferC2[z][i][j] < 0)
						bufferC2[z][i][j] = 0.0;
					//Hongyi Wang for debugging
					//System.out.print(bufferC2[z][i][j]);
				}
		}
		
		stride = 2;
		size = (size - MP2KernalSize)/stride + 1;
		// Hongyi Wang print size for debugging;
		 System.out.print(size);

		max = Double.NEGATIVE_INFINITY;
		Double bufferMP2[][][] = new Double[numMP2][size][size];

		for (int z = 0; z < numMP2; z++) {
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					max = Double.NEGATIVE_INFINITY;
					for (int k = 0; k < MP2KernalSize; k++) {
						for (int l = 0; l < MP2KernalSize; l++) {
							if (bufferC2[z][i * stride + k][j * stride + l] > max) {
								max = bufferC2[z][i * stride + k][j * stride + l];
							}
						}
						bufferMP1[z][i][j] = max;
						// Hongyi Wang print max for debugging;
						// System.out.print(max);
					}

				}
			}
		}
		
		
		return rst;
	}
}
