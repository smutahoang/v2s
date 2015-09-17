package hoang.v2s.factorization;

public class Tweet {
	public String tweetId;
	public Topic[] topTopics;

	public void topicNormalization() {
		double sumProb = 0;
		for (int i = 0; i < topTopics.length; i++) {
			sumProb += topTopics[i].topicProb;
		}
		for (int i = 0; i < topTopics.length; i++) {
			topTopics[i].topicProb /= sumProb;
		}
	}
}
