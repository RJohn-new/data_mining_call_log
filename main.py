import pandas as pd
from apyori import apriori

# Create File object for the rules
File = open('Output/no_priority_rules.txt', 'w')


# Use Pandas to Read the file into a data frame
def read_data():
    df = pd.read_csv('data/Calls.csv').drop('Arrived Time', axis=1).dropna()
    df = df[df['Precinct'] != 'UNKNOWN']
    df = df[df['Priority'] != -1]
    return df[df['Priority'] != 9]


# Split Time Queued into separate Date and Time columns
def split_date_time(df):
    # Create Date Column and cut to only month
    df['Date'] = pd.to_datetime(df['Original Time Queued']).dt.date
    df['Date'] = pd.DatetimeIndex(df['Date']).month
    # Use calendar dictionary to make months standard string values
    import calendar
    month_dict = dict(enumerate(calendar.month_abbr))
    df['Date'] = df['Date'].map(month_dict)

    # Create Time Column and cut to only hour
    df['Time'] = pd.to_datetime(df['Original Time Queued']).dt.time
    df['Time'] = pd.DatetimeIndex(df['Time'].astype(str)).hour

    # Return data frame without the original datetime column
    return df.drop('Original Time Queued', axis=1)


# Cutting times into 5 categories
def categorize_times(df):
    bin_set = [-1, 6, 12, 17, 20, 24]
    label_set = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
    df['Time'] = pd.cut(df['Time'], bins=bin_set, labels=label_set)
    return df


# Use Apyori to mine apriori for rule set
def rule_mine(df):
    df = df.drop('Sector', axis=1).drop('Precinct', axis=1).drop('Priority', axis=1)
    df = df.astype(str)
    record_list = df.values

    assoc_rules = apriori(record_list, min_support=0.005, min_confidence=0.8, min_length=2)

    for item in assoc_rules:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        File.write("Rule: " + items[0] + " -> " + items[1] + '\n')

        # third index of the list located at 0th
        # of the third index of the inner list
        File.write("Confidence: " + "{:.2f}%".format(item[2][0][2] * 100) + '\n')
        File.write("Support: " + "{:.2f}%".format(item[1] * 100) + '\n')
        File.write("Lift: " + "{:.2f}".format(item[2][0][3]) + '\n')
        File.write("==================================================================\n")


# Main method to call other functions
def main():
    df = read_data()
    df = split_date_time(df)
    df = categorize_times(df)
    rule_mine(df)


# Launch main
if __name__ == '__main__':
    main()
