import pandas as pd
import sklearn
import joblib
import numpy as np

# TODO
# add logging
# look at test size, n_estimators, learning rate
# needs tuning
# underperforming in smaller classes, need for data, address imbalance
# ease of use

def training(df):

    print(df)
    # Drop the column with team name
    # No textual identifier in function and can use team id
    team_rankings_df = df.drop("TEAM", axis=1)
    print(team_rankings_df)

    # get shape
    # rows
    print("num_samples", team_rankings_df.shape[0])
    # columns
    print("num_features", team_rankings_df.shape[1] - 1)

    # split the training and test datasets
    # 80% for training and 20% for testing
    train_df, test_df = sklearn.model_selection.train_test_split(
        team_rankings_df,
        test_size=.2
    )

    # What we are trying to predict
    # Get team result in tournament (round)
    y_train = train_df.pop("ROUND")

    # Convert df to an array
    X_train = train_df.values

    # Get team result in tournament (round)
    y_test = test_df.pop("ROUND")

    # Convert df to an array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=100 , learning_rate=.25
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    print(sklearn.metrics.classification_report(
        y_test, 
        y_pred,
        zero_division=0
        ))
    
    joblib.dump(clf, "model.joblib")




def get_team(id):
    # read text file and return team name from id

    with open('id_to_team.txt', "r") as f:
        for line in f:
            # itterate through line and create list from tab delimiter
            line_list = line.split("\t")
            # found id return team
            if int(line_list[0]) == id:
                return line_list[1]

def get_id(team):
    # read text file and return team name from id

    with open('id_to_team.txt', "r") as f:
        for line in f:
            # itterate through line and create list from tab delimiter
            line_list = line.split("\t")
            # found id return team
            if line_list[1].strip("\n") == team:
                return line_list[0]


if __name__=="__main__":

    # make dataframe from csv
    df = pd.read_csv('./TeamRankings.csv', index_col=0)

    # train and store model
    training(df)

    # test it out, I'll use current data for filling out bracket
    X_create = np.array([df.drop("TEAM", axis=1).drop("ROUND", axis=1).values[2]])

    # get saved job
    clf = joblib.load("model.joblib")

    y_pred = clf.predict(X_create)
    print(y_pred)
