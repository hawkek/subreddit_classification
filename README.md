# Subreddit Classification using NLP
## Author: Kirsty Hawke

### Purpose:
Using Pushshift's API, posts from r/science and r/worldnews will be used to train a classifier on which subreddit a given post came from. This binary classification problem aims to illuminate the difference in general news reporting and scientific reporting. The past few years have highlighted the need for unbiased, clear journalism that does sensationalise stories for clickbait. We are all affected by what we read and irresponsible reporting can deepen divisions between people. Success for this project will be a +85% accuracy at classsifying posts correctly.

Additionally, this will be an interesting way to analyze the different community members of each subreddit and how they communicate or respond to articles. Both subreddits post links and not the full articles.

The posts and comments are selected based on their scores, an attibute that conveys importance or agreement. How scoring works: "When you're logged in to Reddit, you'll be able to upvote and downvote items to help determine their rank. You get one vote per item, but you can change it after it's logged. The number appearing between the up and down arrows is the submission's score: the number of upvotes minus the number of downvotes."

### Background Research:
https://www.vox.com/science-and-health/2019/6/11/18652225/hype-science-press-releases

https://www.vox.com/science-and-health/2017/3/3/14792174/half-scientific-studies-news-are-wrong

https://journalofethics.ama-assn.org/article/media-miss-key-points-scientific-reporting/2007-03


### Contents

- A Jupyter Notebook with analysis (https://git.generalassemb.ly/kirstyh/subreddit_classification/blob/master/subreddit_analysis.ipynb)
- An executive summary of the results in a pdf presentation (https://git.generalassemb.ly/kirstyh/subreddit_classification/blob/master/Subreddit_Classification_Model_Pres.pdf)

### Analysis:

The following models were considered in the initial model evaluation:

* Logistic Regression
* K Nearest Neighbours (KNN)
* Multinomial Naïve Bayes
* Random Forest
* Extra Trees
* AdaBoost
* Support-vector Classification

Each model was also evaluated with CountVectorizer and TFIDFVectorizer. 

### Results

| Model                   | Vectorizer | Post Score | Comments Score | Both Score |
| ----------------------- | ---------- | ---------- | -------------- | ---------- |
| Logistic Regression     | Count      | 0.9104     | 0.8071         | 0.8570     |
| Logistic Regression     | TFIDF      | 0.8961     | 0.7859     | 0.8451         |
| Multinomial Naïve Bayes | Count      | 0.9141     | 0.8177         | 0.8616     |
| Multinomial Naïve Bayes| TFIDF                   | 0.9128     | 0.8219     | 0.8628         |
| KNN                     | Count      | 0.6225     | 0.5048         | 0.5981     |
| KNN                     | TFIDF                   | 0.7585     | 0.7404     | 0.5893         |
| Random Forest           | Count      | 0.8871     | 0.7893         | 0.8459     |
| Random Forest           | TFIDF                   | 0.8881     | 0.7845     | 0.8403         |
| Extra Trees             | Count      | 0.9027     | 0.8064         | 0.8547     |
| Extra Trees             | TFIDF                   | 0.9033     | 0.8011     | 0.8562         |
| AdaBoost                | Count      | 0.7769     | 0.7041         | 0.7230     |
| AdaBoost| TFIDF                   | 0.7869     | 0.6967     | 0.7269         |
| SVC                     | Count      | 0.8857     | 0.7419         | 0.8231     |
| SVC| TFIDF                   | 0.9151     | 0.8104     | 0.8659         |
|                         |            |            |                |            |
|                         | Max Score  | 0.9151     | 0.8219         | 0.8659     |

From these results, the multinomial naive bayes model and the SVM model were chosen. Then the hyperparamters were tuned using Grid Search. 

The best models are as follows.

Results Table:

| Model    | Vectorizor | Parameters | CV Score | Train Score | Test Score | Accuracy |
| -------- | ---------- | ---------- | -------- | ----------- | ---------- | --------- |
| MNB      | Count      | analyzer = lemmed, max_df: 0.95, max_features: 20000, stop_words: None, alpha: 1|0.9141  |  0.952 |  0.9208  | 92.1% |
| SVC | TFIDF   |C = 0.9, gamma = 2,analyzer = stemmed_snow, max_df = 0.95, max_features = 10000, min_df = 10| 0.9151   |  0.9898 | 0.926 | 92.6% |


### Evaluation & Recommendation
Compared to the baseline score of 0.5, these two models both exceed that and meet the original goal from the problem statement of having a classifier above 85% accurate. Clearly, the two subreddits are distinguishable by the language used and topics focussed on. 

Both models have a fairly low bias, however both still have a variance over 2%. While the multinomial naive bayes has a higher bias, it does have a lower variance. Depending on the needs of the user, either model would be suitable.

From the confusion matrix, SVC had a more even distribution of false posisitves and false negatives thus I have chosen SVC with TFIDF vectorizing to be the selected one. This model could help journalists sift out common ways on prasing science reporting versus general reporting. Additionally this method can be applied to other subreddits with possibly different models working better for different communities. 

In the future, this project could be further tuned to give better results with better computation power. Additionally, exploring a neural network for this classification problem could proove beneficial.










