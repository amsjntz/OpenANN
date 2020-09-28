package framework;

import java.util.Arrays;
import java.util.Random;

/**
 * @author amsjntz
 */
public class NeuralNetwork {

	public final int inputs, outputs, hiddenlayers, neuronsperlayer;

	public final double[] weights;
	public final double[] biases;

	public boolean useSoftmax = false;

	// fully connected Neural Network
	public NeuralNetwork(int inputs, int outputs, int hiddenlayers, int neuronsperlayer) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.hiddenlayers = hiddenlayers;
		this.neuronsperlayer = neuronsperlayer;
		weights = new double[(inputs * neuronsperlayer) + (neuronsperlayer * neuronsperlayer * (hiddenlayers - 1))
				+ (neuronsperlayer * outputs)];
		randomizeArray(weights, 1, -1);
		biases = new double[hiddenlayers * neuronsperlayer + outputs];
		randomizeArray(biases, 1, -1);
	}

	// copy NeuralNetwork
	public NeuralNetwork(NeuralNetwork nn) {
		this.inputs = nn.inputs;
		this.outputs = nn.outputs;
		this.hiddenlayers = nn.hiddenlayers;
		this.neuronsperlayer = nn.neuronsperlayer;
		this.weights = new double[nn.weights.length];
		System.arraycopy(nn.weights, 0, weights, 0, weights.length);
		this.biases = new double[nn.biases.length];
		System.arraycopy(nn.biases, 0, biases, 0, biases.length);
	}

	private void randomizeArray(double[] src, double min, double max) {
		Random r = new Random();
		for (int i = 0; i < src.length; i++) {
			src[i] = r.nextDouble() * (max * 2) - (-min);
		}
	}
	
	// call this function right after network initialization if you want to redefine the weights
	public void initWeightsInRange(double min, double max) {
		randomizeArray(weights, min, max);
	}
	
	// call this function right after network initialization if you want to redefine the biases
	public void initBiasesInRange(double min, double max) {
		randomizeArray(biases, min, max);
	}
	
	// feed through inputs; returns calculated outputs
	public double[] processData(double[] in) {
		// feeding values from input to first layer
		double[] firstlayer = new double[neuronsperlayer];
		for (int i = 0; i < inputs; i++) {
			for (int j = 0; j < neuronsperlayer; j++) {
				int weightindex = i * neuronsperlayer + j;
				firstlayer[j] += in[i] * weights[weightindex];
			}
		}
		applyBiasesAndActivation(firstlayer, 0);
		// feeding values from first layer through the next layers
		double[] lastlayer = firstlayer; // last calculated layer
		int windadd = inputs * neuronsperlayer; // "offset" of the index in the weights array

		for (int l = 0; l < hiddenlayers - 1; l++) { // hiddenlayers - 1 because first hidden layer has already been
														// calculated
			double[] nextlayer = new double[neuronsperlayer];
			for (int n = 0; n < neuronsperlayer; n++) {
				for (int w = 0; w < neuronsperlayer; w++) {
					int weightindex = windadd + (w * neuronsperlayer) + n;
					nextlayer[n] += lastlayer[w] * weights[weightindex];
				}
			}
			applyBiasesAndActivation(nextlayer, (l + 1) * neuronsperlayer);
			lastlayer = nextlayer;
		}
		// generating output
		windadd = weights.length - (outputs * neuronsperlayer);
		double[] out = new double[outputs];
		for (int n = 0; n < outputs; n++) {
			for (int w = 0; w < neuronsperlayer; w++) {
				int weightindex = windadd + (w * outputs) + n;
				out[n] += lastlayer[w] * weights[weightindex];
			}
		}
		if (useSoftmax) {
			applyBiases(out, biases.length - outputs);
			out = softMax(out);
		} else {
			applyBiasesAndActivation(out, biases.length - outputs);
		}
		return out;
	}

	private void applyBiasesAndActivation(double[] layer, int index) {
		for (int i = 0; i < layer.length; i++) {
			layer[i] += biases[i + index];
			layer[i] = activationFunction(layer[i]);
		}
	}

	private void applyBiases(double[] layer, int index) {
		for (int i = 0; i < layer.length; i++) {
			layer[i] += biases[i + index];
		}
	}

	private double activationFunction(double in) {
		return 1 / (1 + Math.exp(-in)); // sigmoid activation function
	}

	private double[] softMax(double ro[]) {
		double den = 0;
		double[] exps = new double[ro.length];
		for (int i = 0; i < ro.length; i++) {
			exps[i] = Math.exp(ro[i]);
			den += exps[i];
		}
		double[] sm = new double[ro.length];
		for (int i = 0; i < ro.length; i++) {
			sm[i] = exps[i] / den;
		}
		return sm;
	}

	public static double error(double[] actoutput, double[] targetoutput) {
		double error = 0;
		for (int i = 0; i < targetoutput.length; i++) {
			error += Math.pow(actoutput[i] - targetoutput[i], 2);
		}
		error /= targetoutput.length;
		return error;
	}

	public void backpropagate(double[] in, double[] target, double learningrate) {
		double[][] hiddenLayers = new double[hiddenlayers][neuronsperlayer];

		// calculating first hidden layer
		for (int i = 0; i < inputs; i++) {
			for (int j = 0; j < neuronsperlayer; j++) {
				int weightindex = i * neuronsperlayer + j;
				hiddenLayers[0][j] += in[i] * weights[weightindex];
			}
		}
		applyBiasesAndActivation(hiddenLayers[0], 0);

		// feeding values from first layer through the next layers
		double[] lastlayer = hiddenLayers[0]; // last calculated layer
		int windadd = inputs * neuronsperlayer; // "offset" of the index in the weights array

		for (int l = 0; l < hiddenlayers - 1; l++) { // hiddenlayers - 1 because first hidden layer has already been
														// calculated
			double[] nextlayer = new double[neuronsperlayer];
			for (int n = 0; n < neuronsperlayer; n++) {
				for (int w = 0; w < neuronsperlayer; w++) {
					int weightindex = windadd + (w * neuronsperlayer) + n;
					nextlayer[n] += lastlayer[w] * weights[weightindex];
				}
			}
			applyBiasesAndActivation(nextlayer, (l + 1) * neuronsperlayer);
			lastlayer = nextlayer;
			hiddenLayers[l + 1] = lastlayer;
		}

		// generating output
		windadd = weights.length - (outputs * neuronsperlayer);
		double[] out = new double[outputs];
		for (int n = 0; n < outputs; n++) {
			for (int w = 0; w < neuronsperlayer; w++) {
				int weightindex = windadd + (w * outputs) + n;
				out[n] += lastlayer[w] * weights[weightindex];
			}
		}
		if (useSoftmax) {
			applyBiases(out, biases.length - outputs);
			out = softMax(out);
		} else {
			applyBiasesAndActivation(out, biases.length - outputs);
		}

		// output error
		double[] outputError = new double[outputs];
		for (int i = 0; i < out.length; i++) {
			outputError[i] = target[i] - out[i];
		}

		// calculating output gradient
		double[] outgradient = new double[outputs];
		for (int i = 0; i < outgradient.length; i++) {
			outgradient[i] = out[i] * (1 - out[i]) * outputError[i] * learningrate;
		}

		// applying gradient to last layer
		for (int n = 0; n < outputs; n++) {
			for (int w = 0; w < neuronsperlayer; w++) {
				int weightindex = windadd + (w * outputs) + n;
				weights[weightindex] += hiddenLayers[hiddenlayers - 1][w] * outgradient[n];
			}
		}

		// changing output biases
		int biasindex = biases.length - outputs;
		for (int i = 0; i < outputs; i++) {
			biases[biasindex + i] += outgradient[i];
		}

		// applying backpropagation to hidden layers
		double[] recentLayererror = new double[neuronsperlayer];
		for (int i = 0; i < neuronsperlayer; i++) {
			for (int j = 0; j < outputs; j++) {
				recentLayererror[i] += hiddenLayers[hiddenLayers.length - 1][i] * outputError[j];
			}
		}
		for (int i = hiddenlayers - 1; i >= 1; i--) {

			double[] myerrors = new double[neuronsperlayer];
			for (int j = 0; j < neuronsperlayer; j++) {
				for (int k = 0; k < neuronsperlayer; k++) {
					int weightindex = (i + 1) * neuronsperlayer + (k * neuronsperlayer) + j;
					myerrors[j] += recentLayererror[j] * weights[weightindex];
				}
			}

			double[] mygradient = new double[neuronsperlayer];
			for (int g = 0; g < neuronsperlayer; g++) {
				mygradient[g] = hiddenLayers[i][g] * (1 - hiddenLayers[i][g]) * myerrors[g] * learningrate;
			}

			for (int j = 0; j < neuronsperlayer; j++) {
				for (int k = 0; k < neuronsperlayer; k++) {
					int weightindex = (i + 1) * neuronsperlayer + (k * neuronsperlayer) + j;
					weights[weightindex] += hiddenLayers[i][k] * mygradient[k];
				}
			}
			int mybiasindex = i * neuronsperlayer;
			for (int b = 0; b < neuronsperlayer; b++) {
				biases[b + mybiasindex] += mygradient[b];
			}

			recentLayererror = myerrors;
		}

		// backpropagate to the input neurons
		double[] firstlayerErrors = new double[neuronsperlayer];
		for (int j = 0; j < neuronsperlayer; j++) {
			for (int k = 0; k < inputs; k++) {
				int weightindex = (k * neuronsperlayer) + j;
				firstlayerErrors[j] += recentLayererror[j] * weights[weightindex];
			}
		}

		double[] firstlayergradient = new double[neuronsperlayer];
		for (int i = 0; i < neuronsperlayer; i++) {
			firstlayergradient[i] = hiddenLayers[0][i] * (1 - hiddenLayers[0][i]) * firstlayerErrors[i] * learningrate;
		}

		for (int i = 0; i < inputs; i++) {
			for (int j = 0; j < neuronsperlayer; j++) {
				weights[j + (i * neuronsperlayer)] += hiddenLayers[0][j] * firstlayergradient[j];
			}
		}

		for (int i = 0; i < neuronsperlayer; i++) {
			biases[i] += firstlayergradient[i];
		}
	}

	public double[][] getDeltas(double[] in, double[] target) {
		NeuralNetwork copy = new NeuralNetwork(this);
		copy.backpropagate(in, target, 1);

		double[][] allDeltas = new double[2][];
		allDeltas[0] = new double[weights.length];
		allDeltas[1] = new double[biases.length];

		for (int i = 0; i < weights.length; i++) {
			allDeltas[0][i] = copy.weights[i] - weights[i];
		}
		for (int i = 0; i < biases.length; i++) {
			allDeltas[1][i] = copy.biases[i] - biases[i];
		}

		return allDeltas;
	}

	// import NeuralNetwork
	public NeuralNetwork(byte[] bdata) {
		String data = new String(bdata);
		String[] spl = data.split("\r\n");
		String[] attrSpl = spl[0].split(";");
		inputs = Integer.parseInt(attrSpl[0]);
		outputs = Integer.parseInt(attrSpl[1]);
		hiddenlayers = Integer.parseInt(attrSpl[2]);
		neuronsperlayer = Integer.parseInt(attrSpl[3]);

		int totalWeights = (inputs * neuronsperlayer) + (neuronsperlayer * neuronsperlayer * (hiddenlayers - 1))
				+ (neuronsperlayer * outputs);

		weights = new double[totalWeights];
		for (int i = 0; i < totalWeights; i++) {
			weights[i] = Double.parseDouble(spl[i + 1]);
		}

		int totalBiases = hiddenlayers * neuronsperlayer + outputs;
		biases = new double[totalBiases];
		for (int i = 0; i < totalBiases; i++) {
			biases[i] = Double.parseDouble(spl[i + totalWeights + 2]);
		}

		useSoftmax = Boolean.parseBoolean(spl[spl.length - 1]);
	}

	public byte[] exportNeuralNetwork() {
		String out = Integer.toString(inputs) + ";" + Integer.toString(outputs) + ";";
		out += Integer.toString(hiddenlayers) + ";" + Integer.toString(neuronsperlayer) + ";\r\n";
	
		String weightStr = Arrays.toString(weights).substring(1).replace(", ", "\r\n");
		weightStr = weightStr.substring(0, weightStr.length() - 1) + "\r\n";

		out += weightStr + "\r\n";
		
		String biasStr = Arrays.toString(biases).substring(1).replace(", ", "\r\n");
		biasStr = biasStr.substring(0, biasStr.length() - 1) + "\r\n";
		
		out += biasStr;
		
		out += Boolean.toString(useSoftmax);
		return out.getBytes();
	}
}
