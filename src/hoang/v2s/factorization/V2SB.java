package hoang.v2s.factorization;

import hoang.tool.PredictionMetricTool;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class V2SB {
	// data
	public String dataPath;
	public int batch;
	public boolean batchFlag;
	public static int nTopics;
	public static int nTopTopics;
	static RetweetObservation[] trainRTObservations;
	static RetweetObservation[] recTestRTObservations;
	static RetweetObservation[] predTestRTObservations;
	static Tweet[] tweets;
	public boolean learnFlag;
	// scores
	static ViralUser[] viralUsers;
	static SusceptibleUser[] susceptibleUsers;
	static double[] topicViralityScores;

	// popularity
	private static double[] topicTweetingPopularites;
	private static double[] topicRetweetingPopularites;

	// regularization terms
	public static double alphaU;
	public static double alphaT;
	public static double betaU;
	public static double betaT;

	// learning params
	public double lrate_alphaT;
	public double lrate_betaT;

	public double lrate_alphaU;
	public double lrate_betaU;

	public int maxOutterIterations;
	public int maxInnerIterations;
	public int maxFCall;

	// parallel implementation
	public int nParallelThread;

	// output
	public String outputPath;

	// utility variables
	/*
	 * private String trainBatchFileName = "folds/trainingInstances_" + batch +
	 * ".csv"; private String trainFileName =
	 * "selectedAllInOne-ExpandedInstances.txt"; private String
	 * recTestBatchFileName = "folds/testInstances_" + batch + ".csv"; private
	 * String predTestFileName = "ExpandedInstances-LastDay.csv";
	 */
	public String trainBatchFileName;
	public String trainFileName;
	public String recTestBatchFileName;
	public String predTestFileName;
	public String tweetFileName;

	static int[][] topicRelatedRetweetObservations;
	static int[][] topicIndexInRelatedRetweetObservations;

	static double[] topicGrads;
	static double sumTopicViralityScore;
	static double topicDifference;
	static double topicViralityNorm;
	static double[] currTopicViralityScores;
	static double[] newTopicViralityScores;

	private static double[][] viralUserGrads;
	private static double[] sumUserViralityScores;
	static double viralUserDifference;
	static double viralUserNorm;
	static double[][] currUserViralityScores;
	static double[][] newUserViralityScores;
	static int[][] viralUserActiveTopics;
	static public HashMap<Integer, HashMap<Integer, Integer>> viralUserActiveTopicMaps;
	static int[][][] viralUserRelatedRetweetObservationByTopic;

	private static double[][] susceptibleUserGrads;
	private static double[] sumUserSusceptibilityScores;
	static double susceptibleUserDifference;
	static double susceptibleUserNorm;
	static double[][] currUserSusceptibilityScores;
	static double[][] newUserSusceptibilityScores;
	static int[][] susceptibleUserActiveTopics;
	static public HashMap<Integer, HashMap<Integer, Integer>> susceptibleUserActiveTopicMaps;
	static int[][][] susceptibleUserRelatedRetweetObservationByTopic;

	static double[] predValues;
	static int[] threadStartIndexes;
	static int[] threadEndIndexes;
	static double[] threadLikelihood;

	HashMap<String, Integer> tweetId2Index;
	HashMap<String, Integer> senderId2Index;
	HashMap<String, Integer> receiverId2Index;

	static double transform(double z) {
		/*
		 * if (z >= 10) return 1; if (z <= -10) return 0;
		 */
		double val = Math.exp(2 * z);
		val = (val - 1) / (val + 1);
		val = val * 0.5 + 0.5;
		return val;
	}

	static double diffTransform(double z) {
		/*if (z >= 10 || z <= -10)
			return 0;*/
		double val = Math.exp(2 * z);
		val = (2 * val / Math.pow(val + 1, 2));
		return val;
	}

	static class ChildThread implements Runnable {
		private int uIndex;
		private int vIndex;
		private int topicIndex;
		private int threadIndex;
		private boolean topicFlag;
		private boolean viralUserFlag;
		private boolean susceptibleUserFlag;
		private String runOption;

		public ChildThread(int t, int u, int v, boolean tFlag, boolean vUFlag,
				boolean sUFlag, String s) {
			this.topicIndex = t;
			this.uIndex = u;
			this.vIndex = v;
			this.topicFlag = tFlag;
			this.viralUserFlag = vUFlag;
			this.susceptibleUserFlag = sUFlag;
			runOption = s;
		}

		public ChildThread(int threadIndex, boolean tFlag, boolean vUFlag,
				boolean sUFlag, String s) {
			this.threadIndex = threadIndex;
			this.topicFlag = tFlag;
			this.viralUserFlag = vUFlag;
			this.susceptibleUserFlag = sUFlag;
			runOption = s;
		}

		@Override
		public void run() {
			if (runOption.equals("pred")) {
				computePredValue();
			} else if (runOption.equals("tgrad")) {
				computeTopicGrad();
			} else if (runOption.equals("ugrad")) {
				computeViralUserGrad();
			} else if (runOption.equals("vgrad")) {
				computeSusceptibleUserGrad();
			} else {

			}

		}

		private void computePredValue() {
			threadLikelihood[threadIndex] = 0;
			for (int o = threadStartIndexes[threadIndex]; o < threadEndIndexes[threadIndex]; o++) {
				int u = trainRTObservations[o].senderIndex;
				int v = trainRTObservations[o].receiverIndex;
				int m = trainRTObservations[o].tweetIndex;
				predValues[o] = 0;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					double p = tweets[m].topTopics[i].topicProb;

					if (topicFlag) {
						p *= transform(newTopicViralityScores[k]);
					} else {
						p *= transform(currTopicViralityScores[k]);
					}

					if (viralUserFlag) {
						p *= transform(newUserViralityScores[u][k]);
					} else {
						p *= transform(currUserViralityScores[u][k]);
					}

					if (susceptibleUserFlag) {
						p *= transform(newUserSusceptibilityScores[v][k]);
					} else {
						p *= transform(currUserSusceptibilityScores[v][k]);
					}
					predValues[o] += p;
				}
				if (trainRTObservations[o].retweetFlag)
					threadLikelihood[threadIndex] += Math.log(predValues[o]);
				else
					threadLikelihood[threadIndex] += Math
							.log(1 - predValues[o]);
			}
		}

		private void computeTopicGrad() {
			topicGrads[topicIndex] = viralTopicPartialDiff(topicIndex,
					topicFlag, viralUserFlag, susceptibleUserFlag);
		}

		private void computeViralUserGrad() {
			for (int z = 0; z < viralUserActiveTopics[uIndex].length; z++) {
				int j = viralUserActiveTopics[uIndex][z];
				viralUserGrads[uIndex][j] = viralUserPartialDiff(uIndex, j,
						topicFlag, viralUserFlag, susceptibleUserFlag);
			}
		}

		private void computeSusceptibleUserGrad() {
			for (int z = 0; z < susceptibleUserActiveTopics[vIndex].length; z++) {
				int j = susceptibleUserActiveTopics[vIndex][z];
				susceptibleUserGrads[vIndex][j] = susceptibleUserPartialDiff(
						vIndex, j, topicFlag, viralUserFlag,
						susceptibleUserFlag);
			}
		}
	}

	private void readData() {
		try {
			// read tweets

			BufferedReader br = new BufferedReader(new FileReader(dataPath
					+ "/" + tweetFileName));
			int nTweets = 0;
			while (br.readLine() != null) {
				nTweets++;
			}
			br.close();
			tweets = new Tweet[nTweets];
			tweetId2Index = new HashMap<String, Integer>();
			br = new BufferedReader(new FileReader(dataPath + "/"
					+ tweetFileName));
			String line = null;
			nTweets = 0;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				tweets[nTweets] = new Tweet();
				tweets[nTweets].tweetId = tokens[0];
				tweets[nTweets].topTopics = new Topic[nTopTopics];
				for (int i = 0; i < nTopTopics; i++) {
					tweets[nTweets].topTopics[i] = new Topic();
					tweets[nTweets].topTopics[i].topicIndex = Integer
							.parseInt(tokens[2 * i + 1]);
					tweets[nTweets].topTopics[i].topicProb = Double
							.parseDouble(tokens[2 * i + 2]);
				}
				tweetId2Index.put(tokens[0], nTweets);
				tweets[nTweets].topicNormalization();
				nTweets++;
			}
			br.close();

			// read training data
			senderId2Index = new HashMap<String, Integer>();
			HashMap<String, Integer> senderNObservations = new HashMap<String, Integer>();
			receiverId2Index = new HashMap<String, Integer>();
			HashMap<String, Integer> receiverNObservations = new HashMap<String, Integer>();
			if (batchFlag) {
				br = new BufferedReader(new FileReader(dataPath + "/"
						+ trainBatchFileName));
			} else {
				br = new BufferedReader(new FileReader(dataPath + "/"
						+ trainFileName));
			}
			int nObservations = 0;
			line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = null;
				if (batchFlag) {
					tokens = line.split(",");
				} else {
					tokens = line.split("\t");
				}
				String uId = tokens[0];
				if (!senderId2Index.containsKey(uId)) {
					senderId2Index.put(uId, senderId2Index.size());
					senderNObservations.put(uId, 1);
				} else {
					int c = 1 + senderNObservations.get(uId);
					senderNObservations.remove(uId);
					senderNObservations.put(uId, c);
				}

				String vId = tokens[2];
				if (!receiverId2Index.containsKey(vId)) {
					receiverId2Index.put(vId, receiverId2Index.size());
					receiverNObservations.put(vId, 1);
				} else {
					int c = 1 + receiverNObservations.get(vId);
					receiverNObservations.remove(vId);
					receiverNObservations.put(vId, c);
				}
				nObservations++;
			}
			br.close();
			trainRTObservations = new RetweetObservation[nObservations];
			viralUsers = new ViralUser[senderId2Index.size()];
			susceptibleUsers = new SusceptibleUser[receiverId2Index.size()];
			Iterator<Map.Entry<String, Integer>> uIter = senderId2Index
					.entrySet().iterator();
			while (uIter.hasNext()) {
				Map.Entry<String, Integer> uPair = uIter.next();
				String uId = uPair.getKey();
				int uIndex = uPair.getValue();
				int uNObservations = senderNObservations.get(uId);
				viralUsers[uIndex] = new ViralUser();
				viralUsers[uIndex].userId = uId;
				viralUsers[uIndex].retweetObservations = new int[uNObservations];
			}

			Iterator<Map.Entry<String, Integer>> vIter = receiverId2Index
					.entrySet().iterator();
			while (vIter.hasNext()) {
				Map.Entry<String, Integer> vPair = vIter.next();
				String vId = vPair.getKey();
				int vIndex = vPair.getValue();
				int vNObservations = receiverNObservations.get(vId);
				susceptibleUsers[vIndex] = new SusceptibleUser();
				susceptibleUsers[vIndex].userId = vId;
				susceptibleUsers[vIndex].retweetObservations = new int[vNObservations];
			}

			senderNObservations = new HashMap<String, Integer>();
			receiverNObservations = new HashMap<String, Integer>();

			if (batchFlag) {
				br = new BufferedReader(new FileReader(dataPath + "/"
						+ trainBatchFileName));
			} else {
				br = new BufferedReader(new FileReader(dataPath + "/"
						+ trainFileName));
			}

			nObservations = 0;
			line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens;
				if (batchFlag) {
					tokens = line.split(",");
				} else {
					tokens = line.split("\t");
				}

				String uId = tokens[0];
				String vId = tokens[2];
				String tId = tokens[1];

				int uIndex = senderId2Index.get(uId);
				int vIndex = receiverId2Index.get(vId);

				trainRTObservations[nObservations] = new RetweetObservation();
				trainRTObservations[nObservations].senderIndex = uIndex;
				trainRTObservations[nObservations].receiverIndex = vIndex;
				trainRTObservations[nObservations].tweetIndex = tweetId2Index
						.get(tId);
				trainRTObservations[nObservations].retweetFlag = tokens[3]
						.equals("1");

				int rtIndex = 0;
				if (senderNObservations.containsKey(uId)) {
					rtIndex = senderNObservations.get(uId);
					senderNObservations.remove(uId);
				}
				senderNObservations.put(uId, rtIndex + 1);
				viralUsers[uIndex].retweetObservations[rtIndex] = nObservations;

				rtIndex = 0;
				if (receiverNObservations.containsKey(vId)) {
					rtIndex = receiverNObservations.get(vId);
					receiverNObservations.remove(vId);
				}
				receiverNObservations.put(vId, rtIndex + 1);
				susceptibleUsers[vIndex].retweetObservations[rtIndex] = nObservations;

				nObservations++;
			}
			br.close();

			// read recovering test data
			if (batchFlag) {
				nObservations = 0;
				br = new BufferedReader(new FileReader(dataPath + "/"
						+ recTestBatchFileName));
				line = null;
				while ((line = br.readLine()) != null) {
					nObservations++;
				}
				br.close();

				recTestRTObservations = new RetweetObservation[nObservations];

				br = new BufferedReader(new FileReader(dataPath + "/"
						+ recTestBatchFileName));
				nObservations = 0;
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");

					String uId = tokens[0];
					String vId = tokens[2];
					String tId = tokens[1];

					int uIndex = senderId2Index.get(uId);
					int vIndex = receiverId2Index.get(vId);

					recTestRTObservations[nObservations] = new RetweetObservation();
					recTestRTObservations[nObservations].senderIndex = uIndex;
					recTestRTObservations[nObservations].receiverIndex = vIndex;
					recTestRTObservations[nObservations].tweetIndex = tweetId2Index
							.get(tId);
					recTestRTObservations[nObservations].retweetFlag = tokens[3]
							.equals("1");

					nObservations++;
				}
				br.close();
			}
			// read predicting test data
			nObservations = 0;
			br = new BufferedReader(new FileReader(dataPath + "/"
					+ predTestFileName));
			line = null;
			while ((line = br.readLine()) != null) {
				nObservations++;
			}
			br.close();

			predTestRTObservations = new RetweetObservation[nObservations];

			br = new BufferedReader(new FileReader(dataPath + "/"
					+ predTestFileName));
			nObservations = 0;
			line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");

				String uId = tokens[0];
				String vId = tokens[2];
				String tId = tokens[1];

				int uIndex = senderId2Index.get(uId);
				int vIndex = receiverId2Index.get(vId);

				predTestRTObservations[nObservations] = new RetweetObservation();
				predTestRTObservations[nObservations].senderIndex = uIndex;
				predTestRTObservations[nObservations].receiverIndex = vIndex;
				predTestRTObservations[nObservations].tweetIndex = tweetId2Index
						.get(tId);
				predTestRTObservations[nObservations].retweetFlag = tokens[3]
						.equals("1");

				nObservations++;
			}
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void readScores() {
		try {
			topicViralityScores = new double[nTopics];
			BufferedReader br = new BufferedReader(new FileReader(outputPath
					+ "/topicVirality.csv"));
			String line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				int topic = Integer.parseInt(tokens[0]);
				double score = Double.parseDouble(tokens[1]);
				topicViralityScores[topic] = score;
			}
			br.close();

			br = new BufferedReader(new FileReader(outputPath
					+ "/userVirality.csv"));
			line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uId = tokens[0];
				int u = senderId2Index.get(uId);
				viralUsers[u].viralityScores = new double[nTopics];
				for (int i = 1; i < tokens.length; i++) {
					double score = Double.parseDouble(tokens[i]);
					viralUsers[u].viralityScores[i - 1] = score;
				}
			}
			br.close();

			br = new BufferedReader(new FileReader(outputPath
					+ "/userSusceptibility.csv"));
			line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String vId = tokens[0];
				int v = receiverId2Index.get(vId);
				susceptibleUsers[v].susceptibilityScores = new double[nTopics];
				for (int i = 1; i < tokens.length; i++) {
					double score = Double.parseDouble(tokens[i]);
					susceptibleUsers[v].susceptibilityScores[i - 1] = score;
				}
			}
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void getThreadIndexes() {
		threadStartIndexes = new int[nParallelThread];
		threadEndIndexes = new int[nParallelThread];
		threadLikelihood = new double[nParallelThread];
		int chunkLength = trainRTObservations.length / nParallelThread;
		for (int i = 0; i < nParallelThread; i++) {
			threadStartIndexes[i] = i * chunkLength;
			threadEndIndexes[i] = threadStartIndexes[i] + chunkLength;
		}
		threadEndIndexes[nParallelThread - 1] = trainRTObservations.length;
	}

	private void getTopicRelatedRetweetObservations() {
		int[] nRROs = new int[nTopics];
		for (int t = 0; t < nTopics; t++) {
			nRROs[t] = 0;
		}
		for (int o = 0; o < trainRTObservations.length; o++) {
			int tIndex = trainRTObservations[o].tweetIndex;
			for (int i = 0; i < nTopTopics; i++) {
				int k = tweets[tIndex].topTopics[i].topicIndex;
				nRROs[k]++;
			}
		}
		topicRelatedRetweetObservations = new int[nTopics][];
		topicIndexInRelatedRetweetObservations = new int[nTopics][];
		for (int t = 0; t < nTopics; t++) {
			topicRelatedRetweetObservations[t] = new int[nRROs[t]];
			topicIndexInRelatedRetweetObservations[t] = new int[nRROs[t]];
		}
		for (int t = 0; t < nTopics; t++) {
			nRROs[t] = 0;
		}
		for (int o = 0; o < trainRTObservations.length; o++) {
			int tIndex = trainRTObservations[o].tweetIndex;
			for (int i = 0; i < nTopTopics; i++) {
				int k = tweets[tIndex].topTopics[i].topicIndex;
				topicRelatedRetweetObservations[k][nRROs[k]] = o;
				topicIndexInRelatedRetweetObservations[k][nRROs[k]] = i;
				nRROs[k]++;
			}
		}
	}

	private void getUserRelatedRetweetObservationByTopic() {
		viralUserRelatedRetweetObservationByTopic = new int[viralUsers.length][][];
		for (int u = 0; u < viralUsers.length; u++) {
			viralUserRelatedRetweetObservationByTopic[u] = new int[viralUserActiveTopics[u].length][];
			int[] nRelatedObservations = new int[viralUserActiveTopics[u].length];
			for (int i = 0; i < nRelatedObservations.length; i++) {
				nRelatedObservations[i] = 0;
			}
			HashMap<Integer, Integer> activeTopics = viralUserActiveTopicMaps
					.get(u);
			for (int rrt = 0; rrt < viralUsers[u].retweetObservations.length; rrt++) {
				int o = viralUsers[u].retweetObservations[rrt];
				int m = trainRTObservations[o].tweetIndex;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					int index = activeTopics.get(k);
					nRelatedObservations[index]++;
				}
			}
			for (int i = 0; i < nRelatedObservations.length; i++) {
				viralUserRelatedRetweetObservationByTopic[u][i] = new int[nRelatedObservations[i]];
				nRelatedObservations[i] = 0;
			}
			for (int rrt = 0; rrt < viralUsers[u].retweetObservations.length; rrt++) {
				int o = viralUsers[u].retweetObservations[rrt];
				int m = trainRTObservations[o].tweetIndex;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					int index = activeTopics.get(k);
					viralUserRelatedRetweetObservationByTopic[u][index][nRelatedObservations[index]] = o;
					nRelatedObservations[index]++;
				}
			}

		}

		susceptibleUserRelatedRetweetObservationByTopic = new int[susceptibleUsers.length][][];
		for (int v = 0; v < susceptibleUsers.length; v++) {
			susceptibleUserRelatedRetweetObservationByTopic[v] = new int[susceptibleUserActiveTopics[v].length][];
			int[] nRelatedObservations = new int[susceptibleUserActiveTopics[v].length];
			for (int i = 0; i < nRelatedObservations.length; i++) {
				nRelatedObservations[i] = 0;
			}
			HashMap<Integer, Integer> activeTopics = susceptibleUserActiveTopicMaps
					.get(v);
			for (int rrt = 0; rrt < susceptibleUsers[v].retweetObservations.length; rrt++) {
				int o = susceptibleUsers[v].retweetObservations[rrt];
				int m = trainRTObservations[o].tweetIndex;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					int index = activeTopics.get(k);
					nRelatedObservations[index]++;
				}
			}
			for (int i = 0; i < nRelatedObservations.length; i++) {
				susceptibleUserRelatedRetweetObservationByTopic[v][i] = new int[nRelatedObservations[i]];
				nRelatedObservations[i] = 0;
			}
			for (int rrt = 0; rrt < susceptibleUsers[v].retweetObservations.length; rrt++) {
				int o = susceptibleUsers[v].retweetObservations[rrt];
				int m = trainRTObservations[o].tweetIndex;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					int index = activeTopics.get(k);
					susceptibleUserRelatedRetweetObservationByTopic[v][index][nRelatedObservations[index]] = o;
					nRelatedObservations[index]++;
				}
			}
		}

	}

	private void computeTopicPopularities() {
		topicTweetingPopularites = new double[nTopics];
		topicRetweetingPopularites = new double[nTopics];
		for (int i = 0; i < nTopics; i++) {
			topicTweetingPopularites[i] = -1;
			topicRetweetingPopularites[i] = -1;
		}
		int nRetweets = 0;
		int nTweets = 0;
		boolean[] tweetMark = new boolean[tweets.length];
		for (int t = 0; t < tweets.length; t++)
			tweetMark[t] = true;
		for (int o = 0; o < trainRTObservations.length; o++) {
			int tIndex = trainRTObservations[o].tweetIndex;
			if (trainRTObservations[o].retweetFlag) {
				nRetweets++;
				for (int i = 0; i < nTopTopics; i++) {
					int k = tweets[tIndex].topTopics[i].topicIndex;
					double p = tweets[tIndex].topTopics[i].topicProb;
					if (topicRetweetingPopularites[k] < 0) {
						topicRetweetingPopularites[k] = p;
					} else {
						topicRetweetingPopularites[k] += p;
					}
				}
			}
			if (tweetMark[tIndex]) {
				tweetMark[tIndex] = false;
				nTweets++;
				for (int i = 0; i < nTopTopics; i++) {
					int k = tweets[tIndex].topTopics[i].topicIndex;
					double p = tweets[tIndex].topTopics[i].topicProb;
					if (topicTweetingPopularites[k] < 0) {
						topicTweetingPopularites[k] = p;
					} else {
						topicTweetingPopularites[k] += p;
					}
				}
			}
		}
		for (int i = 0; i < nTopics; i++) {
			if (topicTweetingPopularites[i] > 0)
				topicTweetingPopularites[i] /= nTweets;
			if (topicRetweetingPopularites[i] > 0)
				topicRetweetingPopularites[i] /= nRetweets;
		}
	}

	private void computeUserTopicPopularities() {
		for (int u = 0; u < viralUsers.length; u++) {
			viralUsers[u].topicTweetingPopularites = new double[nTopics];
			viralUsers[u].topicRetweetingPopularites = new double[nTopics];

			for (int t = 0; t < nTopics; t++) {
				viralUsers[u].topicTweetingPopularites[t] = -1;
				viralUsers[u].topicRetweetingPopularites[t] = -1;
			}

			HashSet<Integer> uTweets = new HashSet<Integer>();
			int uNRetweet = 0;
			for (int i = 0; i < viralUsers[u].retweetObservations.length; i++) {
				int o = viralUsers[u].retweetObservations[i];
				int tIndex = trainRTObservations[o].tweetIndex;
				if (!uTweets.contains(tIndex)) {
					uTweets.add(tIndex);
					for (int j = 0; j < nTopTopics; j++) {
						int k = tweets[tIndex].topTopics[j].topicIndex;
						double p = tweets[tIndex].topTopics[j].topicProb;
						if (viralUsers[u].topicTweetingPopularites[k] < 0) {
							viralUsers[u].topicTweetingPopularites[k] = p;
						} else {
							viralUsers[u].topicTweetingPopularites[k] += p;
						}
					}
				}
				if (trainRTObservations[o].retweetFlag) {
					uNRetweet++;
					for (int j = 0; j < nTopTopics; j++) {
						int k = tweets[tIndex].topTopics[j].topicIndex;
						double p = tweets[tIndex].topTopics[j].topicProb;
						if (viralUsers[u].topicRetweetingPopularites[k] < 0) {
							viralUsers[u].topicRetweetingPopularites[k] = p;
						} else {
							viralUsers[u].topicRetweetingPopularites[k] += p;
						}
					}
				}
			}
			if (uTweets.size() > 0) {
				for (int t = 0; t < nTopics; t++) {
					if (viralUsers[u].topicTweetingPopularites[t] >= 0)
						viralUsers[u].topicTweetingPopularites[t] /= (uTweets
								.size());
				}
			}
			if (uNRetweet > 0) {
				for (int t = 0; t < nTopics; t++) {
					if (viralUsers[u].topicRetweetingPopularites[t] >= 0)
						viralUsers[u].topicRetweetingPopularites[t] /= uNRetweet;
				}
			}
		}

		for (int v = 0; v < susceptibleUsers.length; v++) {
			susceptibleUsers[v].topicReceivingPopularities = new double[nTopics];
			susceptibleUsers[v].topicRetweetingPopularities = new double[nTopics];

			for (int t = 0; t < nTopics; t++) {
				susceptibleUsers[v].topicReceivingPopularities[t] = -1;
				susceptibleUsers[v].topicRetweetingPopularities[t] = -1;
			}

			int vNRecTweet = 0;
			int vNRetweet = 0;
			for (int i = 0; i < susceptibleUsers[v].retweetObservations.length; i++) {
				int o = susceptibleUsers[v].retweetObservations[i];
				int tIndex = trainRTObservations[o].tweetIndex;
				vNRecTweet++;

				for (int j = 0; j < nTopTopics; j++) {
					int k = tweets[tIndex].topTopics[j].topicIndex;
					double p = tweets[tIndex].topTopics[j].topicProb;
					if (susceptibleUsers[v].topicReceivingPopularities[k] < 0) {
						susceptibleUsers[v].topicReceivingPopularities[k] = p;
					} else {
						susceptibleUsers[v].topicReceivingPopularities[k] += p;
					}
				}

				if (trainRTObservations[o].retweetFlag) {
					vNRetweet++;
					for (int j = 0; j < nTopTopics; j++) {
						int k = tweets[tIndex].topTopics[j].topicIndex;
						double p = tweets[tIndex].topTopics[j].topicProb;
						if (susceptibleUsers[v].topicRetweetingPopularities[k] < 0) {
							susceptibleUsers[v].topicRetweetingPopularities[k] = p;
						} else {
							susceptibleUsers[v].topicRetweetingPopularities[k] += p;
						}
					}
				}
			}
			if (vNRecTweet > 0) {
				for (int t = 0; t < nTopics; t++) {
					if (susceptibleUsers[v].topicReceivingPopularities[t] >= 0)
						susceptibleUsers[v].topicReceivingPopularities[t] /= vNRecTweet;
				}
			}
			if (vNRetweet > 0) {
				for (int t = 0; t < nTopics; t++) {
					if (susceptibleUsers[v].topicRetweetingPopularities[t] >= 0)
						susceptibleUsers[v].topicRetweetingPopularities[t] /= vNRetweet;
				}
			}
		}
	}

	private void getUserActiveTopics() {
		viralUserActiveTopics = new int[viralUsers.length][];
		viralUserActiveTopicMaps = new HashMap<Integer, HashMap<Integer, Integer>>();
		for (int u = 0; u < viralUsers.length; u++) {
			int nActiveTopics = 0;
			for (int t = 0; t < nTopics; t++) {
				if (viralUsers[u].topicTweetingPopularites[t] >= 0)
					nActiveTopics++;
			}
			viralUserActiveTopics[u] = new int[nActiveTopics];
			HashMap<Integer, Integer> activeTopicMap = new HashMap<Integer, Integer>();
			int index = 0;
			for (int t = 0; t < nTopics; t++) {
				if (viralUsers[u].topicTweetingPopularites[t] >= 0) {
					viralUserActiveTopics[u][index] = t;
					activeTopicMap.put(t, index);
					index++;
				}
			}
			viralUserActiveTopicMaps.put(u, activeTopicMap);
		}

		susceptibleUserActiveTopics = new int[susceptibleUsers.length][];
		susceptibleUserActiveTopicMaps = new HashMap<Integer, HashMap<Integer, Integer>>();
		for (int v = 0; v < susceptibleUsers.length; v++) {
			int nActiveTopics = 0;
			for (int t = 0; t < nTopics; t++) {
				if (susceptibleUsers[v].topicReceivingPopularities[t] >= 0)
					nActiveTopics++;
			}
			susceptibleUserActiveTopics[v] = new int[nActiveTopics];
			HashMap<Integer, Integer> activeTopicMap = new HashMap<Integer, Integer>();
			int index = 0;
			for (int t = 0; t < nTopics; t++) {
				if (susceptibleUsers[v].topicReceivingPopularities[t] >= 0) {
					susceptibleUserActiveTopics[v][index] = t;
					activeTopicMap.put(t, index);
					index++;
				}
			}
			susceptibleUserActiveTopicMaps.put(v, activeTopicMap);
		}
	}

	private void getScoreSums(boolean initFlag, boolean topicFlag,
			boolean viralUserFlag, boolean susceptibleUserFlag) {
		if (initFlag) {

			sumTopicViralityScore = 0;
			for (int k = 0; k < nTopics; k++) {
				if (topicTweetingPopularites[k] > 0)
					sumTopicViralityScore += transform(currTopicViralityScores[k]);
			}

			sumUserViralityScores = new double[viralUsers.length];
			for (int u = 0; u < viralUsers.length; u++) {
				sumUserViralityScores[u] = 0;
				for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
					int k = viralUserActiveTopics[u][z];
					sumUserViralityScores[u] += transform(currUserViralityScores[u][k]);
				}
			}

			sumUserSusceptibilityScores = new double[susceptibleUsers.length];
			for (int v = 0; v < susceptibleUsers.length; v++) {
				sumUserSusceptibilityScores[v] = 0;
				for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
					int k = susceptibleUserActiveTopics[v][z];
					sumUserSusceptibilityScores[v] += transform(currUserSusceptibilityScores[v][k]);
				}
			}

		} else {

			if (topicFlag) {
				sumTopicViralityScore = 0;
				for (int k = 0; k < nTopics; k++) {
					if (topicTweetingPopularites[k] > 0)
						sumTopicViralityScore += transform(newTopicViralityScores[k]);
				}
			}
			if (viralUserFlag) {
				sumUserViralityScores = new double[viralUsers.length];
				for (int u = 0; u < viralUsers.length; u++) {
					sumUserViralityScores[u] = 0;
					for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
						int k = viralUserActiveTopics[u][z];
						sumUserViralityScores[u] += transform(newUserViralityScores[u][k]);
					}
				}
			}
			if (susceptibleUserFlag) {
				sumUserSusceptibilityScores = new double[susceptibleUsers.length];
				for (int v = 0; v < susceptibleUsers.length; v++) {
					sumUserSusceptibilityScores[v] = 0;
					for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
						int k = susceptibleUserActiveTopics[v][z];
						sumUserSusceptibilityScores[v] += transform(newUserSusceptibilityScores[v][k]);
					}
				}
			}
		}
	}

	private void getPredValues(boolean topicFlag, boolean viralUserFlag,
			boolean susceptibleUserFlag) {

		ExecutorService executor = Executors
				.newFixedThreadPool(nParallelThread);
		for (int i = 0; i < nParallelThread; i++) {
			Runnable worker = new ChildThread(i, topicFlag, viralUserFlag,
					susceptibleUserFlag, "pred");
			executor.execute(worker);
		}

		executor.shutdown();
		while (!executor.isTerminated()) {
		}
	}

	private double getNegLikelihood(boolean initFlag, boolean topicFlag,
			boolean viralUserFlag, boolean susceptibleUserFlag) {
		double L = 0;
		// log likelihood
		/*
		 * for (int o = 0; o < trainRTObservations.length; o++) { if
		 * (trainRTObservations[o].retweetFlag) L += Math.log(predValues[o]);
		 * else L += Math.log(1 - predValues[o]); }
		 */
		for (int i = 0; i < nParallelThread; i++)
			L += threadLikelihood[i];
		L = -L;
		// regularization

		getScoreSums(initFlag, topicFlag, viralUserFlag, susceptibleUserFlag);

		if (initFlag) {

			topicDifference = 0;
			topicViralityNorm = 0;
			for (int i = 0; i < nTopics; i++) {
				if (topicTweetingPopularites[i] > 0) {
					topicDifference = topicDifference
							+ Math.pow(transform(currTopicViralityScores[i])
									- topicRetweetingPopularites[i]
									* sumTopicViralityScore, 2);
					topicViralityNorm += Math.pow(
							transform(currTopicViralityScores[i]), 2);
				}
			}

			viralUserDifference = 0;
			viralUserNorm = 0;
			for (int u = 0; u < viralUsers.length; u++) {
				for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
					int i = viralUserActiveTopics[u][z];

					viralUserDifference = viralUserDifference
							+ Math.pow(
									transform(currUserViralityScores[u][i])
											- viralUsers[u].topicRetweetingPopularites[i]
											* sumUserViralityScores[u], 2);
					viralUserNorm += Math.pow(
							transform(currUserViralityScores[u][i]), 2);
				}
			}

			susceptibleUserDifference = 0;
			susceptibleUserNorm = 0;
			for (int v = 0; v < susceptibleUsers.length; v++) {
				for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
					int i = susceptibleUserActiveTopics[v][z];

					susceptibleUserDifference = susceptibleUserDifference
							+ Math.pow(
									transform(currUserSusceptibilityScores[v][i])
											- susceptibleUsers[v].topicRetweetingPopularities[i]
											* sumUserSusceptibilityScores[v], 2);
					susceptibleUserNorm += Math.pow(
							transform(currUserSusceptibilityScores[v][i]), 2);
				}
			}

			L = L + alphaT * topicDifference + betaT * topicViralityNorm;
			L = L + alphaU * (viralUserDifference + susceptibleUserDifference)
					+ betaU * (viralUserNorm + susceptibleUserNorm);

		} else {
			if (topicFlag) {
				topicDifference = 0;
				topicViralityNorm = 0;
				for (int i = 0; i < nTopics; i++) {
					if (topicTweetingPopularites[i] > 0) {
						topicDifference = topicDifference
								+ Math.pow(transform(newTopicViralityScores[i])
										- topicRetweetingPopularites[i]
										* sumTopicViralityScore, 2);
						topicViralityNorm += Math.pow(
								transform(newTopicViralityScores[i]), 2);
					}
				}
			}
			L = L + alphaT * topicDifference + betaT * topicViralityNorm;

			if (viralUserFlag) {
				viralUserDifference = 0;
				viralUserNorm = 0;
				for (int u = 0; u < viralUsers.length; u++) {
					for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
						int i = viralUserActiveTopics[u][z];

						viralUserDifference = viralUserDifference
								+ Math.pow(
										transform(newUserViralityScores[u][i])
												- viralUsers[u].topicRetweetingPopularites[i]
												* sumUserViralityScores[u], 2);
						viralUserNorm += Math.pow(
								transform(newUserViralityScores[u][i]), 2);
					}
				}
			}

			if (susceptibleUserFlag) {
				susceptibleUserDifference = 0;
				susceptibleUserNorm = 0;
				for (int v = 0; v < susceptibleUsers.length; v++) {
					for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
						int i = susceptibleUserActiveTopics[v][z];

						susceptibleUserDifference = susceptibleUserDifference
								+ Math.pow(
										transform(newUserSusceptibilityScores[v][i])
												- susceptibleUsers[v].topicRetweetingPopularities[i]
												* sumUserSusceptibilityScores[v],
										2);
						susceptibleUserNorm += Math
								.pow(transform(newUserSusceptibilityScores[v][i]),
										2);
					}
				}
			}

			L = L + alphaU * (viralUserDifference + susceptibleUserDifference)
					+ betaU * (viralUserNorm + susceptibleUserNorm);
		}
		return L;
	}

	static double viralTopicPartialDiff(int j, boolean topicFlag,
			boolean vUFlag, boolean sUFlag) {
		if (topicTweetingPopularites[j] < 0)
			return 0;
		double diffL = 0;
		for (int rrt = 0; rrt < topicRelatedRetweetObservations[j].length; rrt++) {
			int o = topicRelatedRetweetObservations[j][rrt];
			int index = topicIndexInRelatedRetweetObservations[j][rrt];
			int u = trainRTObservations[o].senderIndex;
			int v = trainRTObservations[o].receiverIndex;
			int m = trainRTObservations[o].tweetIndex;
			boolean rFlag = trainRTObservations[o].retweetFlag;

			double diffValue = 0;

			diffValue = tweets[m].topTopics[index].topicProb;

			if (topicFlag) {
				diffValue *= diffTransform(newTopicViralityScores[j]);
			} else {
				diffValue *= diffTransform(currTopicViralityScores[j]);
			}

			if (vUFlag) {
				diffValue *= transform(newUserViralityScores[u][j]);
			} else {
				diffValue *= transform(currUserViralityScores[u][j]);
			}

			if (sUFlag) {
				diffValue *= transform(newUserSusceptibilityScores[v][j]);
			} else {
				diffValue *= transform(currUserSusceptibilityScores[v][j]);
			}
			if (rFlag)
				diffValue = diffValue / predValues[o];
			else
				diffValue = (-diffValue) / (1 - predValues[o]);
			diffL += diffValue;
		}
		double d2Popularity = 0;

		for (int i = 0; i < nTopics; i++) {
			if (i == j) {
				if (topicFlag) {
					d2Popularity = d2Popularity
							+ 2
							* (transform(newTopicViralityScores[i]) - topicRetweetingPopularites[i]
									* sumTopicViralityScore)
							* (diffTransform(newTopicViralityScores[i]) - topicRetweetingPopularites[i]
									* diffTransform(newTopicViralityScores[j]));
				} else {
					d2Popularity = d2Popularity
							+ 2
							* (transform(currTopicViralityScores[i]) - topicRetweetingPopularites[i]
									* sumTopicViralityScore)
							* (diffTransform(currTopicViralityScores[i]) - topicRetweetingPopularites[i]
									* diffTransform(currTopicViralityScores[j]));
				}

			} else {
				if (topicTweetingPopularites[i] > 0) {
					if (topicFlag) {
						d2Popularity = d2Popularity
								+ 2
								* (transform(newTopicViralityScores[i]) - topicRetweetingPopularites[i]
										* sumTopicViralityScore)
								* (0 - topicRetweetingPopularites[i]
										* diffTransform(newTopicViralityScores[j]));
					} else {
						d2Popularity = d2Popularity
								+ 2
								* (transform(currTopicViralityScores[i]) - topicRetweetingPopularites[i]
										* sumTopicViralityScore)
								* (0 - topicRetweetingPopularites[i]
										* diffTransform(currTopicViralityScores[j]));
					}
				}
			}
		}

		diffL = diffL + alphaT * d2Popularity;

		double norm;
		if (topicFlag) {
			norm = 2 * transform(newTopicViralityScores[j])
					* diffTransform(newTopicViralityScores[j]);
		} else {
			norm = 2 * transform(currTopicViralityScores[j])
					* diffTransform(currTopicViralityScores[j]);
		}
		diffL = diffL + betaT * norm;

		diffL = -diffL;
		return diffL;
	}

	static double viralUserPartialDiff(int u, int j, boolean topicFlag,
			boolean vUFlag, boolean sUFlag) {
		int index = viralUserActiveTopicMaps.get(u).get(j);
		double diffL = 0;
		for (int rrt = 0; rrt < viralUserRelatedRetweetObservationByTopic[u][index].length; rrt++) {
			int o = viralUserRelatedRetweetObservationByTopic[u][index][rrt];
			int v = trainRTObservations[o].receiverIndex;
			int m = trainRTObservations[o].tweetIndex;
			boolean flag = trainRTObservations[o].retweetFlag;
			double diffValue = 0;
			for (int i = 0; i < tweets[m].topTopics.length; i++) {
				int k = tweets[m].topTopics[i].topicIndex;
				if (k == j) {
					diffValue = tweets[m].topTopics[i].topicProb;
					if (topicFlag) {
						diffValue *= transform(newTopicViralityScores[j]);

					} else {
						diffValue *= transform(currTopicViralityScores[j]);
					}

					if (vUFlag) {
						diffValue *= diffTransform(newUserViralityScores[u][k]);

					} else {
						diffValue *= diffTransform(currUserViralityScores[u][k]);
					}

					if (sUFlag) {
						diffValue *= transform(newUserSusceptibilityScores[v][k]);
					} else {
						diffValue *= transform(currUserSusceptibilityScores[v][k]);
					}

					break;
				}
			}
			if (flag)
				diffValue = diffValue / predValues[o];
			else
				diffValue = (-diffValue) / (1 - predValues[o]);

			diffL = diffL + diffValue;

		}
		double d2Popularity = 0;
		for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
			int i = viralUserActiveTopics[u][z];
			if (i == j) {
				if (vUFlag) {
					d2Popularity = d2Popularity
							+ 2
							* (transform(newUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* sumUserViralityScores[u])
							* (diffTransform(newUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* diffTransform(newUserViralityScores[u][j]));
				} else {
					d2Popularity = d2Popularity
							+ 2
							* (transform(currUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* sumUserViralityScores[u])
							* (diffTransform(currUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* diffTransform(currUserViralityScores[u][j]));
				}
			} else {
				if (vUFlag) {
					d2Popularity = d2Popularity
							+ 2
							* (transform(newUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* sumUserViralityScores[u])
							* (0 - viralUsers[u].topicRetweetingPopularites[i]
									* diffTransform(newUserViralityScores[u][j]));
				} else {
					d2Popularity = d2Popularity
							+ 2
							* (transform(currUserViralityScores[u][i]) - viralUsers[u].topicRetweetingPopularites[i]
									* sumUserViralityScores[u])
							* (0 - viralUsers[u].topicRetweetingPopularites[i]
									* diffTransform(currUserViralityScores[u][j]));
				}
			}
		}

		diffL = diffL + alphaU * d2Popularity;

		double norm;

		if (vUFlag) {
			norm = 2 * transform(newUserViralityScores[u][j])
					* diffTransform(newUserViralityScores[u][j]);
		} else {
			norm = 2 * transform(currUserViralityScores[u][j])
					* diffTransform(currUserViralityScores[u][j]);
		}

		diffL = diffL + betaU * norm;

		diffL = -diffL;

		return diffL;
	}

	static double susceptibleUserPartialDiff(int v, int j, boolean topicFlag,
			boolean vUFlag, boolean sUFlag) {
		int index = susceptibleUserActiveTopicMaps.get(v).get(j);
		double diffL = 0;
		for (int rrt = 0; rrt < susceptibleUserRelatedRetweetObservationByTopic[v][index].length; rrt++) {
			int o = susceptibleUserRelatedRetweetObservationByTopic[v][index][rrt];
			int u = trainRTObservations[o].senderIndex;
			int m = trainRTObservations[o].tweetIndex;
			boolean flag = trainRTObservations[o].retweetFlag;
			double diffValue = 0;
			for (int i = 0; i < tweets[m].topTopics.length; i++) {
				int k = tweets[m].topTopics[i].topicIndex;
				if (k == j) {
					diffValue = tweets[m].topTopics[i].topicProb;
					if (topicFlag) {
						diffValue *= transform(newTopicViralityScores[j]);

					} else {
						diffValue *= transform(currTopicViralityScores[j]);
					}

					if (vUFlag) {
						diffValue *= transform(newUserViralityScores[u][k]);

					} else {
						diffValue *= transform(currUserViralityScores[u][k]);
					}

					if (sUFlag) {
						diffValue *= diffTransform(newUserSusceptibilityScores[v][k]);
					} else {
						diffValue *= diffTransform(currUserSusceptibilityScores[v][k]);
					}
					break;
				}
			}
			if (flag)
				diffValue = diffValue / predValues[o];
			else
				diffValue = (-diffValue) / (1 - predValues[o]);
			diffL = diffL + diffValue;

			if (Double.isInfinite(diffL)) {
				System.out.println("predValues[o] " + predValues[o]);
				System.exit(-1);
			}
		}
		double d2Popularity = 0;

		for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
			int i = susceptibleUserActiveTopics[v][z];
			if (i == j) {
				if (sUFlag) {
					d2Popularity = d2Popularity
							+ 2
							* (transform(newUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* sumUserSusceptibilityScores[v])
							* (diffTransform(newUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* diffTransform(newUserSusceptibilityScores[v][j]));
				} else {
					d2Popularity = d2Popularity
							+ 2
							* (transform(currUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* sumUserSusceptibilityScores[v])
							* (diffTransform(currUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* diffTransform(currUserSusceptibilityScores[v][j]));
				}
			} else {

				if (sUFlag) {
					d2Popularity = d2Popularity
							+ 2
							* (transform(newUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* sumUserSusceptibilityScores[v])
							* (0 - susceptibleUsers[v].topicRetweetingPopularities[i]
									* diffTransform(newUserSusceptibilityScores[v][j]));
				} else {
					d2Popularity = d2Popularity
							+ 2
							* (transform(currUserSusceptibilityScores[v][i]) - susceptibleUsers[v].topicRetweetingPopularities[i]
									* sumUserSusceptibilityScores[v])
							* (0 - susceptibleUsers[v].topicRetweetingPopularities[i]
									* diffTransform(currUserSusceptibilityScores[v][j]));
				}
			}
		}

		diffL = diffL + alphaU * d2Popularity;

		double norm;
		if (sUFlag) {
			norm = 2 * transform(newUserSusceptibilityScores[v][j])
					* diffTransform(newUserSusceptibilityScores[v][j]);
		} else {
			norm = 2 * transform(currUserSusceptibilityScores[v][j])
					* diffTransform(currUserSusceptibilityScores[v][j]);
		}

		diffL = diffL + betaU * norm;

		diffL = -diffL;

		return diffL;
	}

	private void getGradient(boolean topicFlag, boolean viralUserFlag,
			boolean susceptibleUserFlag) {
		if (topicFlag) {
			topicGrads = new double[nTopics];
			ExecutorService executor = Executors
					.newFixedThreadPool(nParallelThread);
			for (int j = 0; j < nTopics; j++) {
				Runnable worker = new ChildThread(j, 0, 0, topicFlag,
						viralUserFlag, susceptibleUserFlag, "tgrad");
				executor.execute(worker);
			}
			executor.shutdown();
			while (!executor.isTerminated()) {
			}
		}

		if (viralUserFlag) {
			viralUserGrads = new double[viralUsers.length][nTopics];

			ExecutorService executor = Executors
					.newFixedThreadPool(nParallelThread);
			for (int u = 0; u < viralUsers.length; u++) {
				Runnable worker = new ChildThread(0, u, 0, topicFlag,
						viralUserFlag, susceptibleUserFlag, "ugrad");
				executor.execute(worker);
			}
			executor.shutdown();
			while (!executor.isTerminated()) {
			}

		}
		if (susceptibleUserFlag) {
			susceptibleUserGrads = new double[susceptibleUsers.length][nTopics];
			ExecutorService executor = Executors
					.newFixedThreadPool(nParallelThread);
			for (int v = 0; v < susceptibleUsers.length; v++) {
				Runnable worker = new ChildThread(0, 0, v, topicFlag,
						viralUserFlag, susceptibleUserFlag, "vgrad");
				executor.execute(worker);
			}
			executor.shutdown();
			while (!executor.isTerminated()) {
			}
		}
	}

	private double getGradNorm(boolean topicFlag, boolean viralUserFlag,
			boolean susceptibleUserFlag) {
		double norm = 0;
		if (topicFlag) {
			for (int j = 0; j < nTopics; j++) {
				norm += Math.pow(topicGrads[j], 2);
			}
		}

		if (viralUserFlag) {
			for (int u = 0; u < viralUsers.length; u++) {
				for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
					int j = viralUserActiveTopics[u][z];
					norm += Math.pow(viralUserGrads[u][j], 2);
				}
			}
		}

		if (susceptibleUserFlag) {
			for (int v = 0; v < susceptibleUsers.length; v++) {
				for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
					int j = susceptibleUserActiveTopics[v][z];
					norm += Math.pow(susceptibleUserGrads[v][j], 2);
				}
			}
		}
		return norm;
	}

	private void getNewUserScores(String vsFlag, double lambda) {
		if (vsFlag.toLowerCase().equals("v")) {
			for (int u = 0; u < viralUsers.length; u++) {
				for (int z = 0; z < viralUserActiveTopics[u].length; z++) {
					int j = viralUserActiveTopics[u][z];
					newUserViralityScores[u][j] = currUserViralityScores[u][j]
							- lambda * viralUserGrads[u][j];
				}
			}
		} else {
			for (int v = 0; v < susceptibleUsers.length; v++) {
				for (int z = 0; z < susceptibleUserActiveTopics[v].length; z++) {
					int j = susceptibleUserActiveTopics[v][z];
					newUserSusceptibilityScores[v][j] = currUserSusceptibilityScores[v][j]
							- lambda * susceptibleUserGrads[v][j];
				}
			}
		}
	}

	private void getNewTopicScores(double lambda) {
		for (int j = 0; j < nTopics; j++) {
			if (topicTweetingPopularites[j] > 0) {
				newTopicViralityScores[j] = currTopicViralityScores[j] - lambda
						* topicGrads[j];
			} else {
				newTopicViralityScores[j] = Double.NaN;
			}
		}
	}

	private void init() {
		// topic virality
		topicViralityScores = new double[nTopics];
		for (int k = 0; k < nTopics; k++) {
			if (topicTweetingPopularites[k] >= 0) {
				double d = topicRetweetingPopularites[k];
				if (d > 0)
					topicViralityScores[k] = 0.5 * Math.log(d / (1 - d));
				else
					topicViralityScores[k] = -10;
			} else {
				topicViralityScores[k] = Double.NaN;
			}
		}
		// user virality
		for (int u = 0; u < viralUsers.length; u++) {
			viralUsers[u].viralityScores = new double[nTopics];
			for (int k = 0; k < nTopics; k++) {
				if (viralUsers[u].topicTweetingPopularites[k] >= 0) {
					double d = viralUsers[u].topicRetweetingPopularites[k];
					if (d > 0)
						viralUsers[u].viralityScores[k] = 0.5 * Math.log(d
								/ (1 - d));
					else
						viralUsers[u].viralityScores[k] = -10;
				} else {
					viralUsers[u].viralityScores[k] = Double.NaN;
				}
			}
		}

		// user susceptibility
		for (int v = 0; v < susceptibleUsers.length; v++) {
			susceptibleUsers[v].susceptibilityScores = new double[nTopics];
			for (int k = 0; k < nTopics; k++) {
				if (susceptibleUsers[v].topicReceivingPopularities[k] >= 0) {
					double d = susceptibleUsers[v].topicRetweetingPopularities[k];
					if (d > 0)
						susceptibleUsers[v].susceptibilityScores[k] = 0.5 * Math
								.log(d / (1 - d));
					else
						susceptibleUsers[v].susceptibilityScores[k] = -10;
				} else {
					susceptibleUsers[v].susceptibilityScores[k] = Double.NaN;
				}
			}
		}
	}

	private void solve_allAL() {

		currTopicViralityScores = new double[nTopics];
		newTopicViralityScores = new double[nTopics];

		currUserViralityScores = new double[viralUsers.length][nTopics];
		newUserViralityScores = new double[viralUsers.length][nTopics];

		currUserSusceptibilityScores = new double[susceptibleUsers.length][nTopics];
		newUserSusceptibilityScores = new double[susceptibleUsers.length][nTopics];

		for (int j = 0; j < nTopics; j++) {
			currTopicViralityScores[j] = topicViralityScores[j];
			for (int u = 0; u < viralUsers.length; u++) {
				currUserViralityScores[u][j] = viralUsers[u].viralityScores[j];
			}
			for (int v = 0; v < susceptibleUsers.length; v++) {
				currUserSusceptibilityScores[v][j] = susceptibleUsers[v].susceptibilityScores[j];
			}
		}
		// outputVector(currTopicViralityScores, "topicTrace-B.csv");

		predValues = new double[trainRTObservations.length];
		getPredValues(false, false, false);

		double currErro = getNegLikelihood(true, false, false, false);

		for (int outter = 0; outter < maxOutterIterations; outter++) {
			// optimize by topic dimensions
			boolean sFlag = false;
			for (int inner = 0; inner < maxInnerIterations; inner++) {
				getGradient(true, false, false);
				double gradNorm = getGradNorm(true, false, false);

				// double lambda = lrate_betaT;
				double lambda = 0.1 / Math.sqrt(gradNorm);
				getNewTopicScores(lambda);

				getPredValues(true, false, false);
				double newErro = getNegLikelihood(false, true, false, false);
				int nFCall = 1;

				boolean flag = false;
				double errDiff = currErro - newErro;
				double bound = lrate_alphaT * lambda * gradNorm;

				if (errDiff <= bound)
					flag = true;

				System.out.println("b-model\ttopic\t" + outter + "\t" + inner
						+ "\t" + nFCall + "\t" + currErro + "\t" + newErro
						+ "\t" + errDiff + "\t" + bound + "\t" + flag + "\t"
						+ gradNorm);

				while (flag) {
					flag = false;
					// lambda *= lrate_betaT;
					lambda *= 0.1;

					getNewTopicScores(lambda);

					getPredValues(true, false, false);

					newErro = getNegLikelihood(false, true, false, false);
					nFCall++;

					errDiff = currErro - newErro;
					bound = lrate_alphaT * lambda * gradNorm;

					if (errDiff <= bound)
						flag = true;

					System.out.println("b-model\ttopic\t" + outter + "\t"
							+ inner + "\t" + nFCall + "\t" + currErro + "\t"
							+ newErro + "\t" + errDiff + "\t" + bound + "\t"
							+ flag + "\t" + gradNorm);

					if (nFCall >= maxFCall)
						break;
				}
				if (nFCall < maxFCall) {
					currErro = newErro;
					for (int j = 0; j < nTopics; j++) {
						currTopicViralityScores[j] = newTopicViralityScores[j];
					}
					sFlag = true;
					// outputVector(currTopicViralityScores,
					// "topicTrace-B.csv");
				}
			}

			if (sFlag) {
				for (int j = 0; j < nTopics; j++) {
					topicViralityScores[j] = currTopicViralityScores[j];
				}
			}

			// optimize by viral user dimensions
			sFlag = false;
			for (int inner = 0; inner < maxInnerIterations; inner++) {
				getGradient(false, true, false);
				double gradNorm = getGradNorm(false, true, false);

				// double lambda = lrate_betaU;
				double lambda = 1 / Math.sqrt(gradNorm);
				getNewUserScores("v", lambda);

				getPredValues(false, true, false);
				double newErro = getNegLikelihood(false, false, true, false);
				int nFCall = 1;

				boolean flag = false;

				double errDiff = currErro - newErro;
				double bound = lrate_alphaT * lambda * gradNorm;

				if (errDiff <= bound)
					flag = true;

				System.out.println("b-model\tviral-user\t" + outter + "\t"
						+ inner + "\t" + nFCall + "\t" + currErro + "\t"
						+ newErro + "\t" + errDiff + "\t" + bound + "\t" + flag
						+ "\t" + gradNorm);

				while (flag) {
					flag = false;
					// lambda *= lrate_betaU;
					lambda *= 0.1;

					getNewUserScores("v", lambda);

					getPredValues(false, true, false);
					newErro = getNegLikelihood(false, false, true, false);
					nFCall++;

					errDiff = currErro - newErro;
					bound = lrate_alphaT * lambda * gradNorm;

					if (errDiff <= bound)
						flag = true;

					System.out.println("b-model\tviral-user\t" + outter + "\t"
							+ inner + "\t" + nFCall + "\t" + currErro + "\t"
							+ newErro + "\t" + errDiff + "\t" + bound + "\t"
							+ flag + "\t" + gradNorm);

					if (nFCall >= maxFCall)
						break;
				}
				if (nFCall < maxFCall) {
					currErro = newErro;
					for (int j = 0; j < nTopics; j++) {
						for (int u = 0; u < viralUsers.length; u++) {
							currUserViralityScores[u][j] = newUserViralityScores[u][j];
						}
					}
					sFlag = true;
				}
			}
			if (sFlag) {
				for (int j = 0; j < nTopics; j++) {
					for (int u = 0; u < viralUsers.length; u++) {
						viralUsers[u].viralityScores[j] = currUserViralityScores[u][j];
					}
				}
			}

			// optimize by susceptible user dimensions
			sFlag = false;
			for (int inner = 0; inner < maxInnerIterations; inner++) {
				getGradient(false, false, true);
				double gradNorm = getGradNorm(false, false, true);

				// double lambda = lrate_betaU;
				double lambda = 1 / Math.sqrt(gradNorm);
				getNewUserScores("s", lambda);

				getPredValues(false, false, true);
				double newErro = getNegLikelihood(false, false, false, true);
				int nFCall = 1;

				boolean flag = false;
				double errDiff = currErro - newErro;
				double bound = lrate_alphaT * lambda * gradNorm;

				if (errDiff <= bound)
					flag = true;

				System.out.println("b-model\tsus-user\t" + outter + "\t"
						+ inner + "\t" + nFCall + "\t" + currErro + "\t"
						+ newErro + "\t" + errDiff + "\t" + bound + "\t" + flag
						+ "\t" + gradNorm);

				while (flag) {
					flag = false;
					// lambda *= lrate_betaU;
					lambda *= 0.1;

					getNewUserScores("s", lambda);

					getPredValues(false, false, true);
					newErro = getNegLikelihood(false, false, false, true);
					nFCall++;

					errDiff = currErro - newErro;
					bound = lrate_alphaT * lambda * gradNorm;

					if (errDiff <= bound)
						flag = true;

					System.out.println("b-model\tsus-user\t" + outter + "\t"
							+ inner + "\t" + nFCall + "\t" + currErro + "\t"
							+ newErro + "\t" + errDiff + "\t" + bound + "\t"
							+ flag + "\t" + gradNorm);

					if (nFCall >= maxFCall)
						break;
				}
				if (nFCall < maxFCall) {
					currErro = newErro;
					for (int j = 0; j < nTopics; j++) {
						for (int v = 0; v < susceptibleUsers.length; v++) {
							currUserSusceptibilityScores[v][j] = newUserSusceptibilityScores[v][j];
						}
					}
					sFlag = true;
				}
			}
			if (sFlag) {
				for (int j = 0; j < nTopics; j++) {
					for (int v = 0; v < susceptibleUsers.length; v++) {
						susceptibleUsers[v].susceptibilityScores[j] = currUserSusceptibilityScores[v][j];
					}
				}
			}
		}
	}

	private void getRuningTime() {

		try {
			currTopicViralityScores = new double[nTopics];
			newTopicViralityScores = new double[nTopics];

			currUserViralityScores = new double[viralUsers.length][nTopics];
			newUserViralityScores = new double[viralUsers.length][nTopics];

			currUserSusceptibilityScores = new double[susceptibleUsers.length][nTopics];
			newUserSusceptibilityScores = new double[susceptibleUsers.length][nTopics];

			for (int j = 0; j < nTopics; j++) {
				currTopicViralityScores[j] = topicViralityScores[j];
				for (int u = 0; u < viralUsers.length; u++) {
					currUserViralityScores[u][j] = viralUsers[u].viralityScores[j];
				}
				for (int v = 0; v < susceptibleUsers.length; v++) {
					currUserSusceptibilityScores[v][j] = susceptibleUsers[v].susceptibilityScores[j];
				}
			}

			predValues = new double[trainRTObservations.length];

			long startTime = System.currentTimeMillis();
			// erro
			getPredValues(false, false, false);
			getNegLikelihood(true, false, false, false);
			// gradient
			getGradient(true, false, false);
			getGradient(false, true, false);
			getGradient(false, false, true);
			// update
			double gradNorm = getGradNorm(true, false, false);
			double lambda = Math.min(0.1 / Math.sqrt(gradNorm), 0.1);
			getNewTopicScores(lambda);

			gradNorm = getGradNorm(false, true, false);
			lambda = Math.min(1 / Math.sqrt(gradNorm), 0.1);
			getNewUserScores("v", lambda);

			gradNorm = getGradNorm(false, false, true);
			lambda = Math.min(1 / Math.sqrt(gradNorm), 0.1);
			getNewUserScores("s", lambda);

			long endTime = System.currentTimeMillis();
			BufferedWriter bw = new BufferedWriter(new FileWriter(
					"/home/tahoang/DiffusionBehavior/output/runningTime.txt",
					true));
			bw.write("b-model," + viralUsers.length + ","
					+ trainRTObservations.length + "," + nParallelThread + ","
					+ nTopics + "," + (endTime - startTime) + "\n");
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private double getTweetRecPerf(boolean recOrPredFlag) {
		if (recOrPredFlag) {
			HashMap<Integer, HashSet<Integer>> receiverTweets = new HashMap<Integer, HashSet<Integer>>();
			for (int o = 0; o < recTestRTObservations.length; o++) {
				if (recTestRTObservations[o].retweetFlag) {
					int v = recTestRTObservations[o].receiverIndex;
					if (receiverTweets.containsKey(v))
						continue;
					receiverTweets.put(v, new HashSet<Integer>());
				}
			}

			for (int o = 0; o < recTestRTObservations.length; o++) {
				int v = recTestRTObservations[o].receiverIndex;
				if (!receiverTweets.containsKey(v))
					continue;
				HashSet<Integer> vTweets = receiverTweets.get(v);
				vTweets.add(o);
				receiverTweets.remove(v);
				receiverTweets.put(v, vTweets);
			}

			double avgAUCPR = 0;
			PredictionMetricTool predTool = new PredictionMetricTool();
			Iterator<Map.Entry<Integer, HashSet<Integer>>> vIter = receiverTweets
					.entrySet().iterator();
			while (vIter.hasNext()) {
				Map.Entry<Integer, HashSet<Integer>> vPair = vIter.next();
				HashSet<Integer> vTweets = vPair.getValue();
				boolean[] labels = new boolean[vTweets.size()];
				double[] scores = new double[vTweets.size()];
				int index = 0;
				Iterator<Integer> tIter = vTweets.iterator();
				while (tIter.hasNext()) {
					int o = tIter.next();

					int u = recTestRTObservations[o].senderIndex;
					int v = recTestRTObservations[o].receiverIndex;
					int m = recTestRTObservations[o].tweetIndex;

					scores[index] = 0;
					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[index] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[index] = recTestRTObservations[o].retweetFlag;
					index++;
				}
				avgAUCPR += predTool.getAUPRC(labels, scores);
			}
			avgAUCPR /= receiverTweets.size();
			return avgAUCPR;
		} else {
			HashMap<Integer, HashSet<Integer>> receiverTweets = new HashMap<Integer, HashSet<Integer>>();
			for (int o = 0; o < predTestRTObservations.length; o++) {
				if (predTestRTObservations[o].retweetFlag) {
					int v = predTestRTObservations[o].receiverIndex;
					if (receiverTweets.containsKey(v))
						continue;
					receiverTweets.put(v, new HashSet<Integer>());
				}
			}

			for (int o = 0; o < predTestRTObservations.length; o++) {
				int v = predTestRTObservations[o].receiverIndex;
				if (!receiverTweets.containsKey(v))
					continue;
				HashSet<Integer> vTweets = receiverTweets.get(v);
				vTweets.add(o);
				receiverTweets.remove(v);
				receiverTweets.put(v, vTweets);
			}

			double avgAUCPR = 0;
			PredictionMetricTool predTool = new PredictionMetricTool();
			Iterator<Map.Entry<Integer, HashSet<Integer>>> vIter = receiverTweets
					.entrySet().iterator();
			while (vIter.hasNext()) {
				Map.Entry<Integer, HashSet<Integer>> vPair = vIter.next();
				HashSet<Integer> vTweets = vPair.getValue();
				boolean[] labels = new boolean[vTweets.size()];
				double[] scores = new double[vTweets.size()];
				int index = 0;
				Iterator<Integer> tIter = vTweets.iterator();
				while (tIter.hasNext()) {
					int o = tIter.next();

					int u = predTestRTObservations[o].senderIndex;
					int v = predTestRTObservations[o].receiverIndex;
					int m = predTestRTObservations[o].tweetIndex;

					scores[index] = 0;
					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[index] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[index] = predTestRTObservations[o].retweetFlag;
					index++;
				}
				avgAUCPR += predTool.getAUPRC(labels, scores);
			}
			avgAUCPR /= receiverTweets.size();
			return avgAUCPR;
		}
	}

	private double getTweetRecPerf_ExcludingNewTweets(boolean recOrPredFlag) {

		HashSet<Integer> subTweets = new HashSet<Integer>();
		for (int o = 0; o < trainRTObservations.length; o++) {
			if (!subTweets.contains(trainRTObservations[o].tweetIndex)) {
				subTweets.add(trainRTObservations[o].tweetIndex);
			}
		}

		if (recOrPredFlag) {
			HashMap<Integer, HashSet<Integer>> receiverTweets = new HashMap<Integer, HashSet<Integer>>();
			for (int o = 0; o < recTestRTObservations.length; o++) {
				if (!subTweets.contains(recTestRTObservations[o].tweetIndex))
					continue;
				if (recTestRTObservations[o].retweetFlag) {
					int v = recTestRTObservations[o].receiverIndex;
					if (receiverTweets.containsKey(v))
						continue;
					receiverTweets.put(v, new HashSet<Integer>());
				}
			}

			for (int o = 0; o < recTestRTObservations.length; o++) {
				if (!subTweets.contains(recTestRTObservations[o].tweetIndex))
					continue;
				int v = recTestRTObservations[o].receiverIndex;
				if (!receiverTweets.containsKey(v))
					continue;
				HashSet<Integer> vTweets = receiverTweets.get(v);
				vTweets.add(o);
				receiverTweets.remove(v);
				receiverTweets.put(v, vTweets);
			}

			double avgAUCPR = 0;
			PredictionMetricTool predTool = new PredictionMetricTool();
			Iterator<Map.Entry<Integer, HashSet<Integer>>> vIter = receiverTweets
					.entrySet().iterator();
			while (vIter.hasNext()) {
				Map.Entry<Integer, HashSet<Integer>> vPair = vIter.next();
				HashSet<Integer> vTweets = vPair.getValue();
				boolean[] labels = new boolean[vTweets.size()];
				double[] scores = new double[vTweets.size()];
				int index = 0;
				Iterator<Integer> tIter = vTweets.iterator();
				while (tIter.hasNext()) {
					int o = tIter.next();

					int u = recTestRTObservations[o].senderIndex;
					int v = recTestRTObservations[o].receiverIndex;
					int m = recTestRTObservations[o].tweetIndex;

					scores[index] = 0;
					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[index] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[index] = recTestRTObservations[o].retweetFlag;
					index++;
				}
				avgAUCPR += predTool.getAUPRC(labels, scores);
			}
			avgAUCPR /= receiverTweets.size();
			return avgAUCPR;
		} else {
			HashMap<Integer, HashSet<Integer>> receiverTweets = new HashMap<Integer, HashSet<Integer>>();
			for (int o = 0; o < predTestRTObservations.length; o++) {
				if (!subTweets.contains(predTestRTObservations[o].tweetIndex))
					continue;
				if (predTestRTObservations[o].retweetFlag) {
					int v = predTestRTObservations[o].receiverIndex;
					if (receiverTweets.containsKey(v))
						continue;
					receiverTweets.put(v, new HashSet<Integer>());
				}
			}

			for (int o = 0; o < predTestRTObservations.length; o++) {
				if (!subTweets.contains(predTestRTObservations[o].tweetIndex))
					continue;
				int v = predTestRTObservations[o].receiverIndex;
				if (!receiverTweets.containsKey(v))
					continue;
				HashSet<Integer> vTweets = receiverTweets.get(v);
				vTweets.add(o);
				receiverTweets.remove(v);
				receiverTweets.put(v, vTweets);
			}

			double avgAUCPR = 0;
			PredictionMetricTool predTool = new PredictionMetricTool();
			Iterator<Map.Entry<Integer, HashSet<Integer>>> vIter = receiverTweets
					.entrySet().iterator();
			while (vIter.hasNext()) {
				Map.Entry<Integer, HashSet<Integer>> vPair = vIter.next();
				HashSet<Integer> vTweets = vPair.getValue();
				boolean[] labels = new boolean[vTweets.size()];
				double[] scores = new double[vTweets.size()];
				int index = 0;
				Iterator<Integer> tIter = vTweets.iterator();
				while (tIter.hasNext()) {
					int o = tIter.next();

					int u = predTestRTObservations[o].senderIndex;
					int v = predTestRTObservations[o].receiverIndex;
					int m = predTestRTObservations[o].tweetIndex;

					scores[index] = 0;
					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[index] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[index] = predTestRTObservations[o].retweetFlag;
					index++;
				}
				avgAUCPR += predTool.getAUPRC(labels, scores);
			}
			avgAUCPR /= receiverTweets.size();
			return avgAUCPR;
		}
	}

	private void outputLearningParameters() {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath
					+ "/parameters-B.txt"));
			bw.write("alphaT = " + alphaT + "\n");
			bw.write("alphaU = " + alphaU + "\n");
			bw.write("betaT = " + betaT + "\n");
			bw.write("betaU = " + betaU + "\n");

			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void outputScores() {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath
					+ "/topicVirality-B.csv"));
			for (int j = 0; j < nTopics; j++)
				// bw.write(j + "," + transform(topicViralityScores[j]) + "\n");
				bw.write(j + "," + topicViralityScores[j] + "\n");
			bw.close();

			bw = new BufferedWriter(new FileWriter(outputPath
					+ "/userVirality-B.csv"));
			for (int u = 0; u < viralUsers.length; u++) {
				bw.write(viralUsers[u].userId);
				for (int j = 0; j < nTopics; j++)
					// bw.write("," +
					// transform(viralUsers[u].viralityScores[j]));
					bw.write("," + viralUsers[u].viralityScores[j]);
				bw.write("\n");
			}
			bw.close();

			bw = new BufferedWriter(new FileWriter(outputPath
					+ "/userSusceptibility-B.csv"));
			for (int v = 0; v < susceptibleUsers.length; v++) {
				bw.write(susceptibleUsers[v].userId);
				for (int j = 0; j < nTopics; j++)
					// bw.write(","
					// +
					// transform(susceptibleUsers[v].susceptibilityScores[j]));
					bw.write("," + susceptibleUsers[v].susceptibilityScores[j]);
				bw.write("\n");
			}
			bw.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void outputErroFragment() {
		try {
			double[] topicErro = new double[nTopics];
			for (int i = 0; i < nTopics; i++)
				topicErro[i] = 0;
			for (int o = 0; o < trainRTObservations.length; o++) {
				int u = trainRTObservations[o].senderIndex;
				int v = trainRTObservations[o].receiverIndex;
				int m = trainRTObservations[o].tweetIndex;
				double predValue = 0;
				for (int i = 0; i < tweets[m].topTopics.length; i++) {
					int k = tweets[m].topTopics[i].topicIndex;
					double p = tweets[m].topTopics[i].topicProb;
					p *= transform(topicViralityScores[k]);
					p *= transform(viralUsers[u].viralityScores[k]);
					p *= transform(susceptibleUsers[v].susceptibilityScores[k]);
					predValue += p;
				}

				double err = Math.log(1 - predValue);
				if (trainRTObservations[o].retweetFlag)
					err = Math.log(predValue);
				int topic = tweets[m].topTopics[0].topicIndex;
				topicErro[topic] += (-err);
			}

			BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath
					+ "/topicError-B.csv"));
			for (int j = 0; j < nTopics; j++)
				bw.write(j + "," + topicErro[j] + "\n");
			bw.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void outputPerformance(boolean recOrPredFlag) {
		try {
			if (recOrPredFlag) {
				BufferedWriter bw = new BufferedWriter(new FileWriter(
						outputPath + "/v2sScores-B_rec.csv"));
				double[] scores = new double[recTestRTObservations.length];
				boolean[] labels = new boolean[recTestRTObservations.length];

				for (int o = 0; o < recTestRTObservations.length; o++) {

					int u = recTestRTObservations[o].senderIndex;
					int v = recTestRTObservations[o].receiverIndex;
					int m = recTestRTObservations[o].tweetIndex;

					scores[o] = 0;

					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[o] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[o] = recTestRTObservations[o].retweetFlag;

					if (recTestRTObservations[o].retweetFlag) {
						bw.write(viralUsers[u].userId + "," + tweets[m].tweetId
								+ "," + susceptibleUsers[v].userId + ",1,"
								+ scores[o] + "\n");
					} else {
						bw.write(viralUsers[u].userId + "," + tweets[m].tweetId
								+ "," + susceptibleUsers[v].userId + ",0,"
								+ scores[o] + "\n");
					}
				}
				bw.close();

				bw = new BufferedWriter(new FileWriter(outputPath
						+ "/v2sAUCPRs-B_rec.csv"));
				PredictionMetricTool predTool = new PredictionMetricTool();
				bw.write(predTool.getAUPRC(labels, scores) + ","
						+ getTweetRecPerf(recOrPredFlag) + ","
						+ getTweetRecPerf_ExcludingNewTweets(recOrPredFlag));
				bw.close();
			} else {
				BufferedWriter bw = new BufferedWriter(new FileWriter(
						outputPath + "/v2sScores-B_pred.csv"));
				double[] scores = new double[predTestRTObservations.length];
				boolean[] labels = new boolean[predTestRTObservations.length];

				for (int o = 0; o < predTestRTObservations.length; o++) {

					int u = predTestRTObservations[o].senderIndex;
					int v = predTestRTObservations[o].receiverIndex;
					int m = predTestRTObservations[o].tweetIndex;

					scores[o] = 0;

					for (int i = 0; i < nTopTopics; i++) {
						int k = tweets[m].topTopics[i].topicIndex;
						if (topicTweetingPopularites[k] < 0
								|| viralUsers[u].topicTweetingPopularites[k] < 0
								|| susceptibleUsers[v].topicReceivingPopularities[k] < 0)
							continue;
						double p = tweets[m].topTopics[i].topicProb;
						scores[o] += (p * transform(topicViralityScores[k])
								* transform(viralUsers[u].viralityScores[k]) * transform(susceptibleUsers[v].susceptibilityScores[k]));

					}
					labels[o] = predTestRTObservations[o].retweetFlag;

					if (predTestRTObservations[o].retweetFlag) {
						bw.write(viralUsers[u].userId + "," + tweets[m].tweetId
								+ "," + susceptibleUsers[v].userId + ",1,"
								+ scores[o] + "\n");
					} else {
						bw.write(viralUsers[u].userId + "," + tweets[m].tweetId
								+ "," + susceptibleUsers[v].userId + ",0,"
								+ scores[o] + "\n");
					}
				}
				bw.close();

				bw = new BufferedWriter(new FileWriter(outputPath
						+ "/v2sAUCPRs-B_pred.csv"));
				PredictionMetricTool predTool = new PredictionMetricTool();
				bw.write(predTool.getAUPRC(labels, scores) + ","
						+ getTweetRecPerf(recOrPredFlag) + ","
						+ getTweetRecPerf_ExcludingNewTweets(recOrPredFlag));
				bw.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public void learn() {
		System.out.print("reading data ... ");
		readData();
		getTopicRelatedRetweetObservations();
		computeTopicPopularities();
		computeUserTopicPopularities();
		getUserActiveTopics();
		getUserRelatedRetweetObservationByTopic();
		getThreadIndexes();
		System.out.println("Done!");
		if (learnFlag) {
			System.out.print("initializing ... ");
			init();
			System.out.println("Done!");

			System.out.println("solving ... ");
			// solve();
			solve_allAL();
			System.out.println("Done!");
		} else {
			System.out.print("reading learnt scores ... ");
			readScores();
			System.out.println("Done!");
		}

		System.out.print("outputing ... ");
		if (learnFlag) {
			outputLearningParameters();
			outputScores();
		}
		if (batchFlag)
			outputPerformance(true);
		outputPerformance(false);
		outputErroFragment();
		System.out.println("Done!");
	}

	public void learnOnly() {
		System.out.print("reading data ... ");
		readData();
		getTopicRelatedRetweetObservations();
		computeTopicPopularities();
		computeUserTopicPopularities();
		getUserActiveTopics();
		getUserRelatedRetweetObservationByTopic();
		getThreadIndexes();
		System.out.println("Done!");

		System.out.print("initializing ... ");
		init();
		System.out.println("Done!");

		System.out.println("solving ... ");
		solve_allAL();
		System.out.println("Done!");

		System.out.print("outputing ... ");
		outputLearningParameters();
		outputScores();
		System.out.println("Done!");
	}

	public void measureComplexity() {
		System.out.print("reading data ... ");
		readData();
		getTopicRelatedRetweetObservations();
		computeTopicPopularities();
		computeUserTopicPopularities();
		getUserActiveTopics();
		getUserRelatedRetweetObservationByTopic();
		getThreadIndexes();
		System.out.println("Done!");
		System.out.print("initializing ... ");
		init();
		System.out.println("Done!");
		System.out.print("getting runing time ... ");
		getRuningTime();
		System.out.println("Done!");

	}
}
