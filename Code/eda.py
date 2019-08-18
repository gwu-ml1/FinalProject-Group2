from common_utils import load_cleaned
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = load_cleaned()
combined = X_train
combined['target'] = y_train
# plt.hist(df.target.apply(lambda x: 1 if x else 0))
# plt.title("Histogram of target variable")
# plt.xlabel('value')
# plt.ylabel('number of occurrences')
# plt.show()
#
# plt.hist(df.innings1_runs)
# plt.title("Histogram of runs scored in 1st inning")
# plt.show()
#
# plt.hist(df.innings1_wickets)
# plt.title("Histogram of wickets from 1st inning")
# plt.show()
#
# plt.hist(df.innings1_overs)
# plt.title("Histogram of inning1_overs")
# plt.show()
#
# plt.hist(df.innings1_balls_bowled)
# plt.title("Histogram of innings1 balls bowled")
# plt.show()

combined.boxplot('date', 'target')
plt.show()

combined.boxplot('innings1_runs', 'target')
plt.show()

combined.boxplot('innings1_wickets', 'target')
plt.show()

combined.boxplot('innings1_overs', 'target')
plt.show()

combined.boxplot('innings1_balls_bowled', 'target')
plt.show()
