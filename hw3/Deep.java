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
	// Double[][][] c2weights;
	Double[][][][] fl1weights;
	Double[][] fl2weights;
	Double[][][][] p1toc2weights;

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

	// Buffers for intermediate sums and max
	Double[][][] bufferC1;
	Double[][][] bufferMP1;
	Double[][][] bufferC2;
	Double[][][] bufferMP2;
	Double[] bufferFL1;
	Double[] bufferFL2;
	Vector<int[]> bufferMaxonMP2;
	Vector<int[]> bufferMaxonMP1;
	Vector<Vector<Double>> trainfeatureVectors;

	// Save the old deltas
	Double[] dc1weights;
	// Double[][][] dc2weights;
	Double[] dfl1weights;
	Double[] dfl2weights;
	Double[] dp1toc2weights;

	public Deep(Vector<Vector<Double>> trainfeatureVectors, Vector<Vector<Double>> tunefeatureVectors,
			Vector<Vector<Double>> testfeatureVectors) {

		this.trainfeatureVectors = trainfeatureVectors;
		// Reading the training set
		// trainingSet = getData(trainfeatureVectors);

		// Randomize the input
		Lab3.permute(trainfeatureVectors);
		Lab3.permute(tunefeatureVectors);
		Lab3.permute(testfeatureVectors);

		// Creating weights arrays
		c1weights = new Double[numC1][C1KernalSize][C1KernalSize];
		p1toc2weights = new Double[numC2][numMP1][C2KernalSize][C2KernalSize];
		// c2weights = new Double[numC2][C2KernalSize][C2KernalSize];
		fl1weights = new Double[numFL][numMP2][5][5];
		fl2weights = new Double[numFL2][numFL];

		// Creating intermediate deltas
		dc1weights = new Double[numC1];
		dp1toc2weights = new Double[numC2];
		// dc2weights = new Double[numC2][C2KernalSize][C2KernalSize];
		dfl1weights = new Double[numFL];
		dfl2weights = new Double[numFL2];

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
		TrainLabel = new Double[TrainSet.length];
		TuneLabel = new Double[TuneSet.length];
		TestLabel = new Double[TestSet.length];

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
		for (int l = 0; l < numC2; l++) {
			for (int i = 0; i < p1toc2weights[0].length; i++) {
				for (int j = 0; j < p1toc2weights[0][0].length; j++) {
					for (int k = 0; k < p1toc2weights[0][0][0].length; k++) {
						try {
							p1toc2weights[l][i][j][k] = r.nextDouble() * 0.01;
						} catch (ArrayIndexOutOfBoundsException e) {
							System.out.println(l + "l");
							System.out.println(i + "i");
							System.out.println(j + "j");
							System.out.println(k + "k");
						}
					}
				}
			}
		}

		// c2weights
		// for (int i = 0; i < c2weights.length; i++) {
		// for (int j = 0; j < c2weights[0].length; j++) {
		// for (int k = 0; k < c2weights[0][0].length; k++) {
		// c2weights[i][j][k] = r.nextDouble() * 0.01;
		// }
		// }
		// }

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
		train();

	}

	// Calculates the output for the network
	public Double[] forward(Double[][] input) {
		// Final result;
		Double[] rst = new Double[this.numFL2];

		stride = 1;
		// input to convolution layer output
		int size = (imageSize - C1KernalSize) / stride + 1;
		// Hongyi Wang print size for debugging;
		// System.out.print(size);

		bufferC1 = new Double[numC1][size][size];

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
		bufferMaxonMP1 = new Vector<int[]>();
		// Hongyi Wang print size for debugging;
		// System.out.print(size);

		double max = Double.NEGATIVE_INFINITY;
		int maxX, maxY;
		maxX = maxY = -1;
		bufferMP1 = new Double[numMP1][size][size];

		for (int z = 0; z < numMP1; z++) {
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					max = Double.NEGATIVE_INFINITY;
					for (int k = 0; k < MP1KernalSize; k++) {
						for (int l = 0; l < MP1KernalSize; l++) {
							if (bufferC1[z][i * stride + k][j * stride + l] > max) {
								max = bufferC1[z][i * stride + k][j * stride + l];
								maxX = i * stride + k;
								maxY = j * stride + l;
							}
						}
						bufferMP1[z][i][j] = max;
						int[] cache = { z, i, j, maxX, maxY };
						bufferMaxonMP1.add(cache);
						// Hongyi Wang print max for debugging;
						// System.out.print(max);
					}

				}
			}
		}

		// mp1 to c2
		stride = 1;
		size = (size - C2KernalSize) / stride + 1;

		// HOngyi Wang for debugging
		// if(size == MP2KernalSize)
		// System.out.print("here");

		double sum = 0;
		bufferC2 = new Double[numC2][size][size];
		for (int z = 0; z < numC2; z++) {

			for (int i = 0; i < size; i++)
				for (int j = 0; j < size; j++) {
					for (int y = 0; y < numMP1; y++) {
						// convolve
						// Get the sub array
						sum = 0;
						Double[][] a = new Double[C2KernalSize][];
						for (int k = 0; k < C2KernalSize; k++) {
							a[k] = Arrays.copyOfRange(bufferMP1[y][i], j, j + C2KernalSize);
						}
						// Convolution
						sum += Util.matrixDot(a, p1toc2weights[z][y]);

					}
					bufferC2[z][i][j] = sum;
					// Bias
					bufferC2[z][i][j] -= biasC2[z];

					// ReLu
					if (bufferC2[z][i][j] < 0)
						bufferC2[z][i][j] = 0.0;
					// Hongyi Wang for debugging
					// System.out.print(bufferC2[z][i][j]);
				}
		}

		// c2 to mp2
		stride = 2;

		// Hongyi Wang for debugging
		double buffer = size;

		size = (size - MP2KernalSize) / stride + 1;
		bufferMaxonMP2 = new Vector<int[]>();

		// Hongyi Wang print size for debugging;
		if (size == 1)
			// System.out.print(buffer);

			max = Double.NEGATIVE_INFINITY;
		bufferMP2 = new Double[numMP2][size][size];
		maxX = -1;
		maxY = maxX;
		for (int z = 0; z < numMP2; z++) {
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					max = Double.NEGATIVE_INFINITY;
					for (int k = 0; k < MP2KernalSize; k++) {
						for (int l = 0; l < MP2KernalSize; l++) {
							if (bufferC2[z][i * stride + k][j * stride + l] > max) {
								max = bufferC2[z][i * stride + k][j * stride + l];
								maxX = i * stride + k;
								maxY = j * stride + l;
							}
						}
						bufferMP2[z][i][j] = max;
						int[] cache = { z, i, j, maxX, maxY };
						bufferMaxonMP2.add(cache);
						// Hongyi Wang print max for debugging;
						// System.out.print(max);
					}

				}
			}
		}

		// mp2 to fullly connect layer
		bufferFL1 = new Double[numFL];
		for (int i = 0; i < bufferFL1.length; i++) {
			sum = 0;
			for (int j = 0; j < numMP2; j++) {
				sum += Util.matrixDot(bufferMP2[j], fl1weights[i][j]);
				// System.out.print (bufferMP2[j]);
			}
			bufferFL1[i] = sum;
			// Bias
			bufferFL1[i] -= biasFL[i];

			// ReLu
			if (bufferFL1[i] < 0)
				bufferFL1[i] = 0.0;
		}

		// Fully connected layer to output
		bufferFL2 = new Double[numFL2];
		for (int i = 0; i < bufferFL2.length; i++) {

			sum = Util.Dot(bufferFL1, fl2weights[i]);

			bufferFL2[i] = sum;
			// Bias
			bufferFL2[i] -= biasFL2[i];

			// Sigmoid
			bufferFL2[i] = sigmoid(bufferFL2[i]);
		}
		rst = bufferFL2;

		 //Hongyi Wang for debugging
		 // for(int i = 0 ; i < bufferFL2.length; i++)
		 // {
		 // System.out.println(rst[i]);
		 // }
		return rst;
	}

	public void train() {
		int maxEpoch = Lab3.maxEpochs;
		int epoch;
		long start = System.currentTimeMillis();
		for (epoch = 0; epoch < maxEpoch; epoch++) {
			// Loop through all trainning samples
			for (int i = 0; i < TrainSet.length; i++) {
				// Forward pass on 1 sample
				Double[] rst = forward(TrainSet[i]);
				// Output to fully connected layer
				double[] buffer = new double[6];
				double[] fl2activationPrime = new double[6];
				double[][] change = new double[numFL2][numFL];
				// Set the correct label to be 1 leave the others to be 0;
				buffer[(int) (TrainLabel[i] * 1)] = 1;
				double[] error = new double[numFL2];
				for (int j = 0; j < numFL2; j++) {
					error[j] = buffer[j] - rst[j];
					fl2activationPrime[j] = sigmoidPrime(rst[j]);
					// Hongyi Wang for debugging
					// System.out.print ("Error: " + error[j] + " ");
					// Calculate weight change for each weight from fl1
					for (int z = 0; z < numFL; z++) {
						dfl2weights[j] = error[j] * fl2activationPrime[j];

						change[j][z] = Lab3.eta * bufferFL1[z] * dfl2weights[j];
						// Hongyi Wang for debugging
						// System.out.print(change[j][z]);
						fl2weights[j][z] += change[j][z];
					}
					biasFL2[j] -= dfl2weights[j] * Lab3.eta;
				}

				// fully connected to mp2
				double sum;
				double[] fl1activationPrime = new double[numFL];
				// For each FL calculate the delta
				for (int z = 0; z < numFL; z++) {
					sum = 0;
					for (int x = 0; x < numFL2; x++) {
						// Sum all the weights times dfl2weights
						sum += fl2weights[x][z] * dfl2weights[x];
					}
					fl1activationPrime[z] = sigmoidPrime(bufferFL1[z]);
					dfl1weights[z] = sum * fl1activationPrime[z];
					// Update weights
					for (int l = 0; l < numMP2; l++) {
						for (int j = 0; j < fl1weights[0][0].length; j++) {
							for (int k = 0; j < fl1weights[0][0].length; j++) {
								double changed = Lab3.eta * bufferMP2[l][j][k] * dfl1weights[z];
								fl1weights[z][l][j][k] += changed;

								biasFL[z] -= Lab3.eta * dfl1weights[z];
								// Hongyi Wang for debugging
								// System.out.println("Change: "+ biasFL[z]);
							}
						}
					}
				}

				// C2 to MP1
				// update only the weights for the max

				// Find the weights that needs to be updated
				for (int[] pos : bufferMaxonMP2) {
					// Calculate delta for weights that needs to be updated
					sum = 0;
					int a = pos[0];
					int b = pos[1];
					int c = pos[2];
					// Add all the delta pointing to the fully connected layer *
					// corresponding weights
					for (int z = 0; z < numFL; z++) {
						sum += fl1weights[z][a][b][c] * dfl1weights[z];
					}
					dp1toc2weights[a] = ReLuPrime(bufferMP2[a][b][c]) * sum;

					// Update weights
					for (int k = 0; k < numC2; k++) {
						for (int x = 0; x < C2KernalSize; x++) {
							for (int y = 0; y < C2KernalSize; y++) {
								// Sum up all the values on MP1 that connects to
								// C2;
								for (int j = 0; j < numMP1; j++) {
									// Update weight on the kernal's each
									// position
									p1toc2weights[a][j][x][y] += Lab3.eta * dp1toc2weights[a]
											* bufferMP1[j][pos[3] + x][pos[4] + y];
								}
							}
						}
					}
				}
				// Update bias
				for (int j = 0; j < numC2; j++) {
					biasC2[j] -= Lab3.eta * dp1toc2weights[j];
				}

				// C1 to input
				// Find the weights that needs to be updated
				for (int[] pos : bufferMaxonMP1) {
					// Calculate delta for weights that needs to be updated
					sum = 0;
					int a = pos[0];
					int b = pos[1];
					int c = pos[2];
					// Add all the delta pointing to the fully connected layer *
					// corresponding weights
					for (int z = 0; z < numC2; z++) {
						// try{
						sum += p1toc2weights[z][a][b%C2KernalSize][c%C2KernalSize] * dp1toc2weights[z];
						// }
						// catch(ArrayIndexOutOfBoundsException e)
						// {
						// System.out.println(z);
						// System.out.println(a);
						// System.out.println(b);
						// System.out.println(c);
						// System.out.println(dp1toc2weights.length);
						// throw new ArrayIndexOutOfBoundsException();
						// }
					}
					dc1weights[a] = ReLuPrime(bufferMP1[a][b][c]) * sum;

					// Update weights
					for (int x = 0; x < C1KernalSize; x++) {
						for (int y = 0; y < C1KernalSize; y++) {
							// Sum up all the values on MP1 that connects to C2;

							// Update weight on the kernal's each position
							c1weights[a][x][y] += Lab3.eta * dc1weights[a] * TrainSet[i][pos[3] + x][pos[4] + y];

						}
					}
				}

				// Update bias
				for (int j = 0; j < numC1; j++) {
					biasC1[j] -= Lab3.eta * dc1weights[j];
				}

			}

			// Check the accuracy on the TuneSet
			double tuneAccuracy = reportAccuracy(TuneSet, TuneLabel);
			// Hongyi Wang for debugging purpose test the Tune Set after each
			// epoch
			System.out.println("Correct tune: " + tuneAccuracy + " ");

			if (tuneAccuracy > 0.5) {

				double accuracy = reportAccuracy(TestSet, TestLabel);
				System.out.println("Done with test accuracy: " + accuracy);
				break;

			}

			System.out.print("Done with epoch: " + epoch + " ");
			System.out.print("Time: " + Lab3.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start));
			start = System.currentTimeMillis();
			// System.out.println(
			// "\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + "
			// of " + comma(tuneFeatureVectors.size())
			// + " (" + truncate((100.0 * best_tuneSetErrors) /
			// tuneFeatureVectors.size(), 2)
			// + "%) at epoch = " + comma(best_epoch) + " (testset errors = " +
			// comma(testSetErrorsAtBestTune)
			// + " of " + comma(testFeatureVectors.size()) + ", "
			// + truncate((100.0 * testSetErrorsAtBestTune) /
			// testFeatureVectors.size(), 2) + "%).\n");

		}

	}

	public double sigmoid(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}

	public double ReLuPrime(double x) {
		return x > 0 ? 1.0 : 0.0;
	}

	public double sigmoidPrime(double output) {
		return output * (1 - output);
	}

	private void reOrder() {
		Lab3.permute(this.trainfeatureVectors);
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
	}

	private double reportAccuracy(Double[][][] set, Double[] setLabel) {
		double correct = 0;
		for (int j = 0; j < set.length; j++) {
			Double[] rst = forward(set[j]);
			double max = Double.NEGATIVE_INFINITY;
			double maxIndex = -1;
			for (int z = 0; z < rst.length; z++) {
				if (rst[z] > max) {
					max = rst[z];
					maxIndex = z;
				}
			}
			if (setLabel[j] == maxIndex)
				correct++;

		}
		return 1.0 * correct / set.length;
	}

}