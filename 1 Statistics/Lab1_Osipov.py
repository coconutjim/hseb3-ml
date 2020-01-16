__author__ = 'Lev Osipov'

import pandas as pd

# Reading data file
df = pd.read_csv('train.csv')
# Total passenger count
size = len(df)

# Task 1
print "Task 1. Sex distribution:\n", df['Sex'].value_counts() / size

# Task 2
print "Task 2. Class distribution\n", df['Pclass'].value_counts(sort=False) / size

# Task 3&4
print "Task 3&4. Mean age dependency:\n", df.groupby(['Sex', 'Pclass'])['Age'].mean()

# Task 5
print "Task 5. Proportion of survived:", float(len(df[df['Survived'] == 1])) / size

# Task 6
print "Task 6. Mean age:", df['Age'].mean(), "\nMean ages grouped by survived:\n", \
    df.groupby('Survived')['Age'].mean()

# Task 7
print "Task 7. Proportion of survived by sex:\n", df[df['Survived'] == 1]['Sex'].value_counts() / \
                                                                  df['Sex'].value_counts()
# Task 8
print "Task 8. Mean ticket price:", df['Fare'].mean(), "\nMean absolute deviation of prices:", \
    df['Fare'].mad()

# Task 9
print "Task 9. Mean ticket price among survived and died:\n", \
    df.groupby('Survived')['Fare'].mean()

# Task 10
print "Task 10. The most popular male name:", df[df['Sex'] == 'male']['Name'].str.extract(
    '(Mr.|Master.|Don.|Rev.) (\w+)')[1].value_counts().idxmax()