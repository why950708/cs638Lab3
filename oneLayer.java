import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import javafx.util.Pair;
import Lab3;

/**
 * Do not modify.
 * 
 * This is the class with the main function
 */

public class oneLayer{
	/**
	 * Runs the tests for HW4
	*/
	public static void init(Vector<Vector<Double>> trainfeatureVectors, Vector<Vector<Double>> tunefeatureVectors,
	Vector<Vector<Double>> testfeatureVectors, int hiddenNodeCount)
	{
		
		
		//Reading the training set 	
		Vector<Instance1> trainingSet=getData(trainfeatureVectors);
		
		
		//Reading the weights
		Double[][] hiddenWeights=new Double[hiddenNodeCount][];
		
		for(int i=0;i<hiddenWeights.length;i++)
		{
			hiddenWeights[i]=new Double[trainingSet.get(0).attributes.size()+1];
		}
		
		Double [][] outputWeights=new Double[trainingSet.get(0).classValues.size()][];
		for (int i=0; i<outputWeights.length; i++) {
			outputWeights[i]=new Double[hiddenWeights.length+1];
		}
		
		readWeights(hiddenWeights,outputWeights);
		
		Double learningRate=Lab3.eta;
		
		if(learningRate>1 || learningRate<=0)
		{
			System.out.println("Incorrect value for learning rate\n");
			System.exit(-1);
		}
		
		NNImpl nn=new NNImpl(trainingSet, Lab3.numberOfHiddenUnits, learningRate, Lab3.maxEpochs,
					hiddenWeights,outputWeights);
		nn.train();
			
		//Reading the testing set 	
		Vector<Instance1> testSet=getData(testfeatureVectors);
			
		Integer[] outputs=new Integer[testSet.size()];
			
			
		int correct=0;
		for(int i=0;i<testSet.size();i++)
		{
			//Getting output from network
			outputs[i]=nn.calculateOutputForInstance(testSet.get(i));
			int actual_idx=-1;
			for (int j=0; j < testSet.get(i).classValues.size(); j++) {
				if (testSet.get(i).classValues.get(j) > 0.5)
					actual_idx=j;
			}
				
			if(outputs[i] == actual_idx)
			{
				correct++;
			} else {
				System.out.println(i + "th instance got an misclassification, expected: " + actual_idx + ". But actual:" + outputs[i]);
			}
		}
			
			System.out.println("Total instances: " + testSet.size());
			System.out.println("Correctly classified: "+correct);
			
	}
		
	// Reads a file and gets the list of instances
	private static Vector<Instance1> getData(Vector<Double> featureVectors)
	{
		Vector<Instance1> data=new Vector<Instance1>();
		//Get attributes
		
		for(Vector<Double> featureVector: featureVectors)
		{
			Instance1 inst = new Instance1();
			for(Vector <Double> attribute : featureVector)
			{	
				//Add the class value 
				if(attribute == featureVector.lastElement())
				{
					for(int i =0; i<6; i++)
					{
						int classVal = (int)attribute;
						if(i != (classVal))
						inst.classValue.add(i,0);
						else
							inst.classValue.add(i,1);
					}
					
				}
				//Add attributes into the instance list
				else
				inst.attributes.add(attribute);
	
			}
			data.add(inst);
	}
		return data;
	}
		
	// Gets weights randomly
	public static void readWeights(Double [][]hiddenWeights, Double[][]outputWeights)
	{
		//Use the given random weight generator in Lab3
		
		for(int i=0;i<hiddenWeights.length;i++)
		{
			for(int j=0;j<hiddenWeights[i].length;j++)
			{	
				//Fan in for the hidden layer is the number of inputs, fan out is the num of outputs
				hiddenWeights[i][j] = Lab3.getRandomWeight(Lab3.inputVectorSize,outputWeights.length);
			}
		}
		
		for(int i=0;i<outputWeights.length;i++)
		{
			for (int j=0; j<outputWeights[i].length; j++)
			{
				//Fan in for the output layer is the number of hidden layer, fan out is 1
				outputWeights[i][j] = Lab3.getRandomWeight(hiddenWeights.length,1);
			}
		
		
		
		//The original method
		// Random r = new Random();
			
		// for(int i=0;i<hiddenWeights.length;i++)
		// {
			// for(int j=0;j<hiddenWeights[i].length;j++)
			// {
				// hiddenWeights[i][j] = r.nextDouble()*0.01;
			// }
		// }
				
		// for(int i=0;i<outputWeights.length;i++)
		// {
			// for (int j=0; j<outputWeights[i].length; j++)
			// {
				// outputWeights[i][j] = r.nextDouble()*0.01;
			// }
		// }	
	

	// }
		}
	}
/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */




class NNImpl {
	public ArrayList<Node> inputNodes = null;// list of the output layer nodes.
	public ArrayList<Node> hiddenNodes = null;// list of the hidden layer nodes
	public ArrayList<Node> outputNodes = null;// list of the output layer nodes

	public ArrayList<Instance1> trainingSet = null;// the training set

	Double learningRate = 1.0; // variable to store the learning rate
	int maxEpoch = 1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network Also
	 * connects the nodes of different layers After calling the constructor the
	 * last node of both inputNodes and hiddenNodes will be bias nodes.
	 */

	public NNImpl(ArrayList<Instance1> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch,
			Double[][] hiddenWeights, Double[][] outputWeights) {
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;

		// input layer nodes
		inputNodes = new ArrayList<Node>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for (int i = 0; i < inputNodeCount; i++) {
			Node node = new Node(0);
			inputNodes.add(node);
		}

		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		// hidden layer nodes
		hiddenNodes = new ArrayList<Node>();
		for (int i = 0; i < hiddenNodeCount; i++) {
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for (int j = 0; j < inputNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		// Output node layer
		outputNodes = new ArrayList<Node>();
		for (int i = 0; i < outputNodeCount; i++) {
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}
			outputNodes.add(node);
		}
	}

	/**
	 * Get the output from the neural network for a single instance Return the
	 * idx with highest output values. For example if the outputs of the
	 * outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs of the
	 * outputNodes are [0.1, 0.5, 0.5], it should return 2. The parameter is a
	 * single instance.
	 */

	public int calculateOutputForInstance(Instance1 inst) {
		// TODO: add code here
		// Get the input from the instance
		for (int i = 0; i < inst.attributes.size(); i++) {
			this.inputNodes.get(i).setInput(inst.attributes.get(i));
		}

		for (Node hiddenNode : this.hiddenNodes) {
			hiddenNode.calculateOutput();
		}

		for (Node outputNode : this.outputNodes) {
			outputNode.calculateOutput();
		}

		double max = 0;
		int index = 0;

		for (int i = 0; i < this.outputNodes.size(); i++) {
			Node output = this.outputNodes.get(i);
			if (output.getOutput() > max) {
				max = output.getOutput();
				index = i;
			}
		}
		return index;

	}

	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train() {
		double sum = 0;
		// TODO: add code here
		for (int i = 0; i < maxEpoch; i++) {

			for (Instance1 inst : this.trainingSet) {
				this.calculateOutputForInstance(inst); // forward
				// Backward
				for (int j = 0; j < this.outputNodes.size(); j++) {
					int derivative = (outputNodes.get(j).getOutput() > 0) ? 1 : 0;
					double TO = inst.classValues.get(j) - outputNodes.get(j).getOutput();

					// Compute Wjk
					
					for (NodeWeightPair pair : outputNodes.get(j).parents) {
						double weight = this.learningRate * pair.node.getOutput() * derivative * TO;
						pair.weight+=weight;
					}

				}
				// Compute Wij
				for (int j = 0; j < this.hiddenNodes.size(); j++) {
					double hiddenDerivative = (hiddenNodes.get(j).getSum()>0)?1:0;
					if(hiddenNodes.get(j).parents ==null)
					{
						continue;
						
					}
					for (NodeWeightPair pair : hiddenNodes.get(j).parents) {
						double total = 0;
						for (Node output : this.outputNodes) {
							double TO = inst.classValues.get(this.outputNodes.indexOf(output)) - output.getOutput();
							int derivative = (output.getSum()> 0) ? 1 : 0;
							total += derivative * TO * getHidenPairWeight(this.hiddenNodes.get(j), output).weight;
						}
						Node input = pair.node;
						double weight =this.learningRate*input.getOutput()*hiddenDerivative*total;
						pair.weight +=weight;
					}
				}
			}
		}
	}

	private NodeWeightPair getHidenPairWeight(Node nodeIn, Node output) {
		// TODO Auto-generated method stub
		for (NodeWeightPair pair : output.parents) {
			if (pair.node.equals(nodeIn))
				return pair;
		}
		return null;
	}
}
/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details
 * 
 * Do not modify. 
 */



class Node{
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents=null; //Array List that will contain the parents (including the bias node) with weights if applicable
		 
	private Double inputValue=0.0;
	private Double outputValue=0.0;
	private Double sum=0.0; // sum of wi*xi
	
	//Create a node with a specific type
	public Node(int type)
	{
		if(type>4 || type<0)
		{
			System.out.println("Incorrect value for node type");
			System.exit(1);
			
		}
		else
		{
			this.type=type;
		}
		
		if (type==2 || type==4)
		{
			parents=new ArrayList<NodeWeightPair>();
		}
	}
	
	//For an input node sets the input value which will be the value of a particular attribute
	public void setInput(Double inputValue)
	{
		if(type==0)//If input node
		{
			this.inputValue=inputValue;
		}
	}
	
	/**
	 * Calculate the output of a ReLU node.
	 * You can assume that outputs of the parent nodes have already been calculated
	 * You can get this value by using getOutput()
	 * @param train: the training set
	 */
	public void calculateOutput()
	{
		
		if(type==2 || type==4)//Not an input or bias node
		{
			// TODO: add code here
			double cache =0 ;
			for(NodeWeightPair nwp :this.parents)
			{
				cache+= nwp.node.getOutput()*nwp.weight;
				
			}
			this.sum = cache;
			
			if(cache < 0) cache = 0;
			
			this.outputValue = cache;
		}
	}

	public double getSum() {
		return sum;
	}
	
	//Gets the output value
	public double getOutput()
	{
		
		if(type==0)//Input node
		{
			return inputValue;
		}
		else if(type==1 || type==3)//Bias node
		{
			return 1.00;
		}
		else
		{
			return outputValue;
		}
		
	}
}


/**
 * Class to identfiy connections
 * between different layers.
 * 
 */

class NodeWeightPair{
	public Node node; //The parent node
	public Double weight; //Weight of this connection
	
	//Create an object with a given parent node 
	//and connect weight
	public NodeWeightPair(Node node, Double weight)
	{
		this.node=node;
		this.weight=weight;
	}
}

/**
 * Holds data for a particular instance.
 * Attributes are represented as an ArrayList of Doubles
 * Class labels are represented as an ArrayList of Integers. For example,
 * a 3-class instance will have classValues as [0 1 0] meaning this 
 * instance has class 1.
 * Do not modify
 */
 

class Instance1{
	public Vector<Double> attributes;
	public Integer classValue;
	
	public Instance1()
	{
		attributes=new Vector<Double>();
		classValues=new Vector<Integer>();
	}
	
}
