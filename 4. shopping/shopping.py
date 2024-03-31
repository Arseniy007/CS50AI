import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename: str) -> tuple:
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidences, labels = list(), list()

    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence = list()
            for column_name, data in row.items():
                if column_name in (
                    "Administrative", "Informational", "ProductRelated",
                    "OperatingSystems", "Browser", "Region", "TrafficType"
                ):
                    evidence.append(int(data))
                elif column_name in (
                    "Administrative_Duration", "Informational_Duration", 
                    "ProductRelated_Duration", "BounceRates", "ExitRates",
                    "PageValues", "SpecialDay"
                ):
                    evidence.append(float(data))
                elif column_name == "VisitorType":
                    visitor_type = 0
                    if data == "Returning_Visitor":
                        visitor_type = 1
                    evidence.append(visitor_type)
                elif column_name == "Weekend":
                    weekend = 0
                    if data == "TRUE":
                        weekend = 1
                    evidence.append(weekend)
                elif column_name == "Month":
                    evidence.append(convert_month_to_int(data))
                elif column_name == "Revenue":
                    label = 0
                    if data == "TRUE":
                        label = 1
                    labels.append(label)
            evidences.append(evidence)
    return (evidences, labels)


def convert_month_to_int(month: str) -> int:
    months = {
        "January": 0, "Feb": 1, "Mar": 2,
        "April": 3, "May": 4, "June": 5, "Jul": 6,
        "Aug": 7, "Sep": 8, "Oct": 9,
        "Nov": 10, "Dec": 11
    }
    return months[month]
    

def train_model(evidence: list, labels: list) -> KNeighborsClassifier:
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
    

def evaluate(labels: list, predictions: list) -> tuple:
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Calculate overall number of positive and negative labels
    positive_cases = len([case for case in labels if case == 1])
    negative_cases = len(labels) - positive_cases
    positive_predictions, negative_predictions = 0, 0

    # Check how well predictions match true labels
    for label, prediction in zip(labels, predictions):
        if label == 1:
            if prediction == label:
                positive_predictions += 1
        else:
            if prediction == label:
                negative_predictions += 1

    # Calculate final rates
    sensitivity = float(positive_predictions / positive_cases)
    specificity = float(negative_predictions / negative_cases)
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
