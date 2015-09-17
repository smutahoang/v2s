package hoang.v2s.factorization;

public class ModelRunner {

	private static void V2SFrunner_GetRunningTime(String path,
			String trainFileName, int nTopics, int nThreads) {

		V2SF model = new V2SF();
		model.dataPath = path;
		model.trainFileName = trainFileName;
		model.tweetFileName = "tweets.csv";
		model.predTestFileName = "predInstances.csv";

		model.learnFlag = true;
		model.batchFlag = false;

		V2SF.alphaT = 0.1;
		V2SF.betaT = 0.1;
		V2SF.alphaU = 1.0E-5;
		V2SF.betaU = 1.0E-5;

		model.outputPath = path;

		V2SF.nTopics = nTopics;
		V2SF.nTopTopics = 3;

		model.lrate_alphaT = 0.01;
		model.lrate_betaT = 0.0001;

		model.lrate_alphaU = 0.01;
		model.lrate_betaU = 0.001;

		model.maxFCall = 10;
		model.maxInnerIterations = 20;
		model.maxOutterIterations = 200;

		model.nParallelThreads = nThreads;
		model.measureComplexity();
	}

	private static void V2SFrunner_LearnOnly(String path, String trainFileName,
			int nTopics, int nThreads) {

		V2SF model = new V2SF();
		model.dataPath = path;
		model.trainFileName = trainFileName;
		model.tweetFileName = "tweets.csv";
		model.predTestFileName = "predInstances.csv";

		model.learnFlag = true;
		model.batchFlag = false;

		V2SF.alphaT = 0.1;
		V2SF.betaT = 0.1;
		V2SF.alphaU = 1.0E-5;
		V2SF.betaU = 1.0E-5;

		model.outputPath = path;

		V2SF.nTopics = nTopics;
		V2SF.nTopTopics = 3;

		model.lrate_alphaT = 0.01;
		model.lrate_betaT = 0.0001;

		model.lrate_alphaU = 0.01;
		model.lrate_betaU = 0.001;

		model.maxFCall = 10;
		model.maxInnerIterations = 20;
		model.maxOutterIterations = 50;

		model.nParallelThreads = nThreads;
		model.learnOnly();
	}

	private static void V2SBrunner_GetRunningTime(String path,
			String trainFileName, int nTopics, int nThreads) {

		V2SB model = new V2SB();
		model.dataPath = path;
		model.trainFileName = trainFileName;
		model.tweetFileName = "tweets.csv";
		model.predTestFileName = "predInstances.csv";

		model.learnFlag = true;
		model.batchFlag = false;

		V2SB.alphaT = 0.1;
		V2SB.betaT = 0.1;
		V2SB.alphaU = 1.0E-5;
		V2SB.betaU = 1.0E-5;

		model.outputPath = path;

		V2SB.nTopics = nTopics;
		V2SB.nTopTopics = 3;

		model.lrate_alphaT = 0.01;
		model.lrate_betaT = 0.0001;

		model.lrate_alphaU = 0.01;
		model.lrate_betaU = 0.001;

		model.maxFCall = 2;
		model.maxInnerIterations = 5;
		model.maxOutterIterations = 10;

		model.nParallelThread = nThreads;
		model.measureComplexity();
	}

	private static void V2SBrunner_LearnOnly(String path, String trainFileName,
			int nTopics, int nThreads) {

		V2SB model = new V2SB();
		model.dataPath = path;
		model.trainFileName = trainFileName;
		model.tweetFileName = "tweets.csv";
		model.predTestFileName = "predInstances.csv";

		model.learnFlag = true;
		model.batchFlag = false;

		V2SB.alphaT = 0.1;
		V2SB.betaT = 0.1;
		V2SB.alphaU = 1.0E-5;
		V2SB.betaU = 1.0E-5;

		model.outputPath = path;

		V2SB.nTopics = nTopics;
		V2SB.nTopTopics = 3;

		model.lrate_alphaT = 0.01;
		model.lrate_betaT = 0.0001;

		model.lrate_alphaU = 0.01;
		model.lrate_betaU = 0.001;

		model.maxFCall = 10;
		model.maxInnerIterations = 20;
		model.maxOutterIterations = 50;

		model.nParallelThread = nThreads;
		model.learnOnly();
	}

	private static void BaselineApproach(int window) {
		V2SF model = new V2SF();
		V2SF.nTopics = 80;
		V2SF.nTopTopics = 3;

		model.dataPath = "F:/Users/tahoang/java/DiffusionBehavior/output/us/Windows/"
				+ window;
		model.trainFileName = "selectedAllInOne-ExpandedInstances.txt";
		model.predTestFileName = "ExpandedInstances-LastDay.csv";
		model.tweetFileName = "unifiedTweetTopics.csv";

		model.outputPath = "F:/Users/tahoang/java/DiffusionBehavior/output/us/Windows/"
				+ window + "/output/baselineApproach";

		model.baseLine();
	}

	public static void main(String[] args) {

		if (args[0].equals("tf")) {
			System.out.println("getting time with f-model");
			V2SFrunner_GetRunningTime(args[1], args[2],
					Integer.parseInt(args[3]), Integer.parseInt(args[4]));
		} else if (args[0].equals("tb")) {
			System.out.println("getting time with b-model");
			V2SBrunner_GetRunningTime(args[1], args[2],
					Integer.parseInt(args[3]), Integer.parseInt(args[4]));
		} else if (args[0].equals("af")) {
			System.out.println("learning with f-model");
			V2SFrunner_LearnOnly(args[1], args[2], Integer.parseInt(args[3]),
					Integer.parseInt(args[4]));
		} else if (args[0].equals("ab")) {
			System.out.println("learning with b-model");
			V2SBrunner_LearnOnly(args[1], args[2], Integer.parseInt(args[3]),
					Integer.parseInt(args[4]));
		} else if (args[0].equals("lda")) {
			System.out.println("learning with baseline");
			BaselineApproach(Integer.parseInt(args[1]));
		} else {
		}
	}
}
