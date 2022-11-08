{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exercise 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For this exercise , you will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set and very often is a student's first step in Data Analytics! \n",
    "\n",
    "The Dataset has been given to you on D2L. You need to download the .csv file from your assignment folder. The above link is just for a reference story about the data. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "1- For this assignment, you need to perform explorotary data analysis and answer at least three hypotheses based on the dataset. You may need to use your knowledge of statiscts to analyze this data.\n",
    "\n",
    "Here are three possible hypotheses that you can define for this dataset (you can define your own hypotheses as well):\n",
    "\n",
    "- Determine if the survival rate is associated to the class of passenger\n",
    "- Determine if the survival rate is associated to the gender\n",
    "- Determine the survival rate is associated to the age\n",
    "\n",
    "\n",
    "\n",
    "2- For each hypothesis, you need to make at least one plot. \n",
    "\n",
    "3- Write a summary of your findings in one page (e.g., summary statistics, plots) and submit the pdf file. Therefore, for part 2 of your assignment, you need to submit one jupyter notebook file and one pdf file.\n",
    "\n",
    "This will be your first end to end data analysis project. For this assignment, you will be graded on you overall analysis, and your final report.\n",
    "\n",
    "4- Push your code and project to github and provide the link to your code here.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ensure that your github project is organized to at least couple of main folders, ensure that you have the README file as well:\n",
    "\n",
    "- Src\n",
    "- Data\n",
    "- Docs\n",
    "- Results\n",
    "\n",
    "Read this link for further info:  https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as pl\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "titanicdf = pd.read_csv(\"titanic.csv\")  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanicdf.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Determine if the survival rate is associated to the class of passenger:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "total = len(titanicdf['Pclass'])\n",
    "first = (titanicdf['Pclass'] == 1).sum()\n",
    "second = (titanicdf['Pclass'] == 2).sum()\n",
    "third = (titanicdf['Pclass'] == 3).sum()\n",
    "\n",
    "first_class =  first / total * 100\n",
    "second_class = second / total * 100\n",
    "third_class = third / total * 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First Class Survival with 216 passengers has a rate of 24.24\n",
      "Second Class Survival rate is 184 passengers has a rate of 20.65\n",
      "Third Class Survival rate is 491 passengers has a rate of 55.11\n"
     ]
    }
   ],
   "source": [
    "print(\"First Class Survival with\", first, \"passengers has a rate of\", round(first_class,2))\n",
    "print(\"Second Class Survival rate is\", second, \"passengers has a rate of\", round(second_class,2))\n",
    "print(\"Third Class Survival rate is\", third, \"passengers has a rate of\", round(third_class,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class</th>\n",
       "      <th>Total Survive</th>\n",
       "      <th>Percentage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>First</td>\n",
       "      <td>216</td>\n",
       "      <td>24.24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Second</td>\n",
       "      <td>184</td>\n",
       "      <td>20.65</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Third</td>\n",
       "      <td>491</td>\n",
       "      <td>55.11</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Class  Total Survive  Percentage\n",
       "0   First            216       24.24\n",
       "1  Second            184       20.65\n",
       "2   Third            491       55.11"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classpercent = [[\"First\",216, 24.24], ['Second', 184, 20.65], ['Third', 491, 55.11]]\n",
    "df2 = pd.DataFrame(classpercent, columns = ['Class', 'Total Survive', 'Percentage'])\n",
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>136</td>\n",
       "      <td>216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>87</td>\n",
       "      <td>184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>119</td>\n",
       "      <td>491</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Survived  Total\n",
       "Pclass                 \n",
       "1            136    216\n",
       "2             87    184\n",
       "3            119    491"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "survivedclassdf = titanicdf[['Survived', 'Pclass']]        \n",
    "survivedclassdf.head()\n",
    "\n",
    "survivepclass = survivedclassdf.groupby(['Pclass']).sum()  \n",
    "totalclass = survivedclassdf.groupby(['Pclass']).count()  #calculating the survived per class\n",
    "\n",
    "totalclass.rename(columns = {'Survived':'Total'}, inplace = True) # Changed column name to Total\n",
    "\n",
    "survivedtotclass = pd.merge(survivepclass, totalclass, left_index=True, right_index=True) # merge by index\n",
    "survivedtotclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count      3.000000\n",
       "mean     114.000000\n",
       "std       24.879711\n",
       "min       87.000000\n",
       "25%      103.000000\n",
       "50%      119.000000\n",
       "75%      127.500000\n",
       "max      136.000000\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "percent_survived = round((survivedtotclass['Survived'] / survivedtotclass['Total']) * 100,2)\n",
    "survivedtotclass['Percentage'] = percent_survived\n",
    "\n",
    "survivedtotclass['Survived'].describe()\n",
    "# the mean value of survived variable per class is 114"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x27926d282b0>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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06ZNm5SSkqJhw4YpMjJSHTt2lCT16NFDYWFhGjp0qHbv3q3169friSeeUEJCgjw8PJx5apVCz549lZGRocOHD2vdunXq0qWLHn74YfXt21dFRUWSfu0B43sFADAzp4ado0eP6u6771bz5s01aNAg1alTRzt27FC9evUkSa+88or69u2rmJgYde7cWVarVatWrbJt7+LiorVr18rFxUWRkZH6+9//rnvvvVfPPPOMs06pUvHw8JDVatV1112ndu3a6fHHH9cHH3ygdevWacGCBZJKX8ZKT0/XoEGD5O/vr4CAAPXr10+HDx+2rS8uLta4cePk7++vOnXq6LHHHpNhGNf2xAAAuAJODTtLly7VsWPHlJ+fr6NHj2rp0qVq2rSpbb2np6dmz56tkydP6ty5c1q1alWpsTiNGjXSJ598ory8PB0/flwvvfRSuc6ebDZdu3ZVmzZt7ELjRYWFhYqOjpaPj4+++OILbdu2Td7e3urZs6ftURnTp0/XggULNG/ePH355Zc6efKkVq9efa1PAwCAMiMVVEOhoaH6/vvvSy1ftmyZSkpK9NZbb9lu254/f778/f21efNm9ejRQzNmzFBiYqIGDBggSZo7d67Wr19/TesHUMUwCwScfAGAsFMNGYZxyTlodu/erUOHDsnHx8du+YULF/Tzzz8rJydHGRkZ6tChg22dq6urIiIiuJQFAKi0CDvV0E8//aQmTZqUWn727FmFh4dr8eLFpdZdHEcFAEBVU6luPUfF+/zzz7Vnzx7FxMSUWteuXTsdPHhQgYGBCgkJsXtdnG26fv36+uqrr2zbFBUVKSUl5VqeAgAAV4SwY2L5+fnKzMzU//73P3377beaPHmy+vXrp759++ree+8t1T42NlZ169ZVv3799MUXXyg1NVWbN2/WQw89pKNHj0qSHn74YU2dOlVr1qzRvn379OCDD5aa+BEAgMqEy1gm9umnn6p+/fpydXVV7dq11aZNG82aNUtxcXGqUaN0zq1Vq5a2bt2qCRMmaMCAATpz5oyuu+46devWzTbL9Pjx45WRkWHbx3333af+/fsrJyfnWp8eAABlYjEYWarc3Fz5+fkpJyen1KMjLly4oNTUVDVp0kSenp5OqhCO4GcHVBLcjYUKShp/9Pf7t7iMBQAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wg3K3efNmWSyWCn9mVnx8vO68884KPQYAoOoj7DjIco1fjjh+/LhGjhyp4OBgeXh4yGq1Kjo6Wtu2bXNwj2Xz17/+VRkZGfLz86vQ4wAAUBY8CNTEYmJiVFBQoIULF+qGG25QVlaWkpKSdOLECYf2ZxiGiouL5er6xx8bd3d3Wa1Wh44BAEB5o2fHpE6fPq0vvvhCL7zwgrp06aJGjRrp5ptvVmJiou644w4dPnxYFotFu3btstvGYrFo8+bNkv7/ctS6desUHh4uDw8PzZs3TxaLRfv27bM73iuvvKKmTZvabXf69Gnl5uaqZs2aWrdunV371atXy8fHR3l5eZKk9PR0DRo0SP7+/goICFC/fv10+PBhW/vi4mKNGzdO/v7+qlOnjh577DHxDFsAQFkQdkzK29tb3t7eWrNmjfLz869qX//61780depU/fTTTxo4cKAiIiK0ePFiuzaLFy/WPffcU2pbX19f9e3bV0uWLCnV/s4771StWrVUWFio6Oho+fj46IsvvtC2bdvk7e2tnj17qqCgQJI0ffp0LViwQPPmzdOXX36pkydPavXq1Vd1XgCA6oGwY1Kurq5asGCBFi5cKH9/f91yyy16/PHH9f3331/xvp555hl1795dTZs2VUBAgGJjY/Xee+/Z1h84cEApKSmKjY295PaxsbFas2aNrRcnNzdXH3/8sa39smXLVFJSorfeekutWrVSixYtNH/+fKWlpdl6mWbMmKHExEQNGDBALVq00Ny5cxkTBAAoE8KOicXExOjYsWP68MMP1bNnT23evFnt2rXTggULrmg/ERERdu+HDBmiw4cPa8eOHZJ+7aVp166dQkNDL7l979695ebmpg8//FCS9P7778vX11dRUVGSpN27d+vQoUPy8fGx9UgFBATowoUL+vnnn5WTk6OMjAx16NDBtk9XV9dSdQEAcCmEHZPz9PRU9+7d9eSTT2r79u2Kj4/XxIkTVaPGrz/63457KSwsvOQ+vLy87N5brVZ17drVdmlqyZIll+3VkX4dsDxw4EC79oMHD7YNdD579qzCw8O1a9cuu9eBAwcueWkMAIArQdipZsLCwnTu3DnVq1dPkpSRkWFb99vByn8mNjZWy5YtU3Jysv773/9qyJAhf9r+008/1d69e/X555/bhaN27drp4MGDCgwM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6VItX75cvXv3Lu96AAAAypVDd2O5u7srJCSkvGsBAAAodw6FnfHjx2vmzJnigekAAKCyc+gy1pdffqlNmzZp3bp1atmypdzc3OzWr1q1qlyKAwAAuFoOhR1/f3/179+/vGsBAAAodw6Fnfnz55d3HQAAABXCoTE70q8P/fzss8/0xhtv2B74eezYMZ09e7bcigMAALhaDvXsHDlyRD179lRaWpry8/PVvXt3+fj46IUXXlB+fr7mzp1b3nUCAAA4xKGenYcfflgRERE6deqUatasaVvev39/JSUllVtxAAAAV8uhnp0vvvhC27dvl7u7u93yxo0b63//+1+5FAYAAFAeHOrZKSkpUXFxcanlR48elY+Pz1UXBQAAUF4cCjs9evTQjBkzbO8tFovOnj2riRMn8ggJAABQqVgMB6ZBPnr0qKKjo2UYhg4ePKiIiAgdPHhQdevW1datWxUYGFgRtVaY3Nxc+fn5KScnR76+vs4uBwDMhWdFo4IeuFDWv98OhR3p11vPly1bpt27d+vs2bNq166dYmNj7QYsVxWEHQCoQIQdVNWwYyaEHQCoQIQdODnsODRmZ+HChfr4449t7x977DH5+/vrr3/9q44cOeLILgEAACqEQ2Fn8uTJtstVycnJeu211zRt2jTVrVtXY8eOLdcCAQAAroZD8+ykp6crJCREkrRmzRoNHDhQDzzwgG655Rbddttt5VkfAADAVXGoZ8fb21snTpyQJG3YsEHdu3eXJHl6eur8+fPlVx0AAMBVcqhnp3v37vrHP/6hm266SQcOHLDNrbN37141bty4POsDAAC4Kg717MyePVuRkZE6fvy43n//fdWpU0eSlJKSorvvvrtcCwQAALga3Houbj0HgArFreeoireef/rpp/ryyy9t72fPnq22bdvqnnvu0alTpxzZJQAAQIVwKOw8+uijys3NlSTt2bNH48ePV+/evZWamqpx48aVa4EAAABXw6EByqmpqQoLC5Mkvf/+++rbt68mT56sb7/9lgeBAgCASsWhnh13d3fl5eVJkj777DP16NFDkhQQEGDr8QEAAKgMHOrZ6dSpk8aNG6dbbrlFX3/9tZYtWyZJOnDggK6//vpyLRAAAOBqONSz89prr8nV1VUrV67U66+/ruuuu06StG7dOvXs2bNcCwQAALga3Houbj0HgArFredw8q3nDl3G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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "x = survivedtotclass.index.values\n",
    "ht = survivedtotclass.Total\n",
    "hs = survivedtotclass.Survived\n",
    "\n",
    "pht = pl.bar(x, ht, color=['magenta'])\n",
    "phs = pl.bar(x, hs, color=['cyan'])\n",
    "\n",
    "pl.xticks(x, x)\n",
    "pl.xlabel('Pclass')\n",
    "pl.ylabel('Passengers')\n",
    "pl.title('Survivors by Class')\n",
    "\n",
    "pl.legend([pht,phs],['Died', 'Survived'])\n",
    "\n",
    "# Displaying the graph of Died and Survived per class, we could see that mostly who died are those from the Pclass 3\n",
    "# and the highest survivoal rate are from Pclass 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As seen in the graph and also from the dataframe table - 1st Class passengers had highest rate of survival, followed by the 3rd class passengers, and the least survival rates was of 3rd class passengers. The passengers travelling in 3rd class were 491 which has the highest number yet only 24.24% survived."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###  Determine if the survival rate is associated to the gender\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "female    233\n",
       "male      109\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "female    314\n",
       "male      577\n",
       "Name: Total, dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>233</td>\n",
       "      <td>314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>109</td>\n",
       "      <td>577</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Survived  Total\n",
       "Sex                    \n",
       "female       233    314\n",
       "male         109    577"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "groupsex = titanicdf.groupby('Sex')\n",
    "\n",
    "survivedsex = groupsex['Survived'].sum()\n",
    "survivedsex.name = 'Survived'\n",
    "display(survivedsex)\n",
    "\n",
    "totalsex = groupsex['Survived'].size()\n",
    "totalsex.name = 'Total'\n",
    "display(totalsex)\n",
    "\n",
    "survivetotalsex = pd.concat([survivedsex, totalsex], axis=1)\n",
    "survivetotalsex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "      <th>Percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>233</td>\n",
       "      <td>314</td>\n",
       "      <td>74.203822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>109</td>\n",
       "      <td>577</td>\n",
       "      <td>18.890815</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Survived  Total  Percentage\n",
       "Sex                                \n",
       "female       233    314   74.203822\n",
       "male         109    577   18.890815"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "percent_survived = (survivetotalsex['Survived'] / survivetotalsex['Total']) * 100\n",
    "survivetotalsex['Percentage'] = percent_survived\n",
    "\n",
    "survivetotalsex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x27928f6cd30>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAHHCAYAAABZbpmkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy89olMNAAAACXBIWXMAAA9hAAAPYQGoP6dpAABDOUlEQVR4nO3df3xP9f//8ftrv9mP18yPzcRMU6z8CMVKKo35mZj8aAmJ95uhSLzXuzfpB9LbO4S3Isb7zTuhvMMbye+YH00klV9hk/0QbcPYz/P9w3evT682Yjavl9Ptermcy8V5nuc553G2Xq/dO+d5zrEYhmEIAADApFwcXQAAAEB5IuwAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAuG79+vVT7dq1HV3GDdm8ebMsFouWLVvm6FIcymKx6LXXXnN0GYBDEHYAJ3XgwAF1795dISEh8vLyUo0aNdSmTRu99957ji4N1/DNN9+of//+Cg0NlZeXl3x8fNS4cWONHj1aP/74o6PLA/6Q3BxdAIDiduzYoccee0y1atXSwIEDFRQUpOTkZO3cuVPTpk3TsGHDHFLXnDlzVFhY6JB93w7mzJmjwYMHq0qVKoqJiVG9evWUn5+vb7/9VgsXLtTUqVN16dIlubq6OrpU4A+FsAM4obfeektWq1V79uyRv7+/3bL09PQy28/Fixfl7e193f3d3d3LbN/XKz8/X4WFhfLw8Ljl+74RO3bs0ODBg/XQQw9p1apV8vX1tVs+ZcoUvfXWWw6qrmxdvnxZHh4ecnHh4gBuD/yXCjihY8eO6Z577ikWdCSpWrVqtn+fOHFCFotF8fHxxfr9dozGa6+9JovFou+++05PP/20KlWqpJYtW+rvf/+7LBaLTp48WWwbcXFx8vDw0C+//CLJfsxOXl6eAgIC1L9//2LrZWVlycvLS6NGjbK1paena8CAAQoMDJSXl5caNWqkBQsW2K1XdDx///vfNXXqVN15553y9PTUd999J0l67733dM8996hixYqqVKmSmjVrpsWLF1/15/hrBQUFeuWVVxQUFCRvb2898cQTSk5Oti0fN26c3N3ddebMmWLrDho0SP7+/rp8+fJVtz9+/HhZLBYtWrSoWNCRJC8vL73xxhvFzurs2rVL7dq1k9VqVcWKFfXII49o+/btdn2KfndHjx5Vv3795O/vL6vVqv79+ys7O9uub05OjkaMGKGqVavK19dXTzzxhE6dOlVizT/99JOee+45BQYGytPTU/fcc4/mzZtn16dozNNHH32kV199VTVq1FDFihWVlZV11Z8F4GwIO4ATCgkJUWJior799tsy3/ZTTz2l7OxsTZgwQQMHDlSPHj1ksVj08ccfF+v78ccfq23btqpUqVKxZe7u7uratatWrFih3Nxcu2UrVqxQTk6OevXqJUm6dOmSHn30Uf3rX/9STEyM3nnnHVmtVvXr10/Tpk0rtu358+frvffe06BBgzRlyhQFBARozpw5Gj58uMLDwzV16lSNHz9ejRs31q5du67ruN966y2tXr1aY8aM0fDhw7V+/XpFRkbq0qVLkqQ+ffooPz9fS5YssVsvNzdXy5YtU3R0tLy8vErcdnZ2tjZu3KhHH31Ud9xxx3XVI0kbN25Uq1atlJWVpXHjxmnChAnKyMhQ69attXv37mL9e/ToofPnz2vixInq0aOH4uPjNX78eLs+zz//vKZOnaq2bdtq0qRJcnd3V8eOHYttKy0tTS1atNAXX3yhoUOHatq0aQoLC9OAAQM0derUYv3feOMNrV69WqNGjdKECROc/kwbYMcA4HQ+//xzw9XV1XB1dTUiIiKM0aNHG+vWrTNyc3Pt+h0/ftyQZMyfP7/YNiQZ48aNs82PGzfOkGT07t27WN+IiAijadOmdm27d+82JBkLFy60tfXt29cICQmxza9bt86QZKxcudJu3Q4dOhh16tSxzU+dOtWQZPz73/+2teXm5hoRERGGj4+PkZWVZXc8fn5+Rnp6ut02u3TpYtxzzz3Fav89mzZtMiQZNWrUsO3HMAzj448/NiQZ06ZNs/s5NG/e3G79Tz75xJBkbNq06ar72L9/vyHJePHFF4stO3v2rHHmzBnblJOTYxiGYRQWFhp169Y1oqKijMLCQlv/7OxsIzQ01GjTpo2treh399xzz9ltu2vXrkblypVt8/v27TMkGUOGDLHr9/TTTxf772HAgAFG9erVjZ9//tmub69evQyr1WpkZ2cbhvF/P786derY2oDbDWd2ACfUpk0bJSQk6IknntD+/fs1efJkRUVFqUaNGvrss89uatt//vOfi7X17NlTiYmJOnbsmK1tyZIl8vT0VJcuXa66rdatW6tKlSp2Z0N++eUXrV+/Xj179rS1/e9//1NQUJB69+5ta3N3d9fw4cN14cIFbdmyxW670dHRqlq1ql2bv7+/Tp06pT179lz/wf7Ks88+a3d5qXv37qpevbr+97//2fXZtWuX3c9h0aJFqlmzph555JGrbrvoko6Pj0+xZXXq1FHVqlVtU9Hvb9++fTpy5IiefvppnT17Vj///LN+/vlnXbx4UY8//ri2bt1abDD4b393Dz/8sM6ePWvbf9GxDB8+3K7fiy++aDdvGIaWL1+uzp07yzAM275//vlnRUVFKTMzU3v37rVbp2/fvqpQocJVfwaAMyPsAE7q/vvv1yeffKJffvlFu3fvVlxcnM6fP6/u3bvbxrCURmhoaLG2p556Si4uLrbQYhiGli5dqvbt28vPz++q23Jzc1N0dLT++9//KicnR5L0ySefKC8vzy7snDx5UnXr1i02oLV+/fq25b9X45gxY+Tj46MHHnhAdevWVWxsbLGxLddSt25du3mLxaKwsDCdOHHC1tazZ095enpq0aJFkqTMzEytWrVKMTExslgsV912UYi6cOFCsWX//e9/tX79ev3973+3az9y5IikKyHi12GoatWqmjt3rnJycpSZmWm3Tq1atezmiy4vFo2pOnnypFxcXHTnnXfa9bv77rvt5s+cOaOMjAx98MEHxfZdNAbrtwPhS/qdALcL7sYCnJyHh4fuv/9+3X///brrrrvUv39/LV26VOPGjbvqH+CCgoKrbq+k/zsPDg7Www8/rI8//livvPKKdu7cqaSkJL399tu/W1+vXr30/vvva82aNXryySf18ccfq169emrUqNH1H+R11Fi/fn0dOnRIq1at0tq1a7V8+XLNmjVLY8eOLTZupbQqVaqkTp06adGiRRo7dqyWLVumnJwcPfPMM9dcLywsTG5ubiWOsSo6I+TmZv91W3TW5p133lHjxo1L3O5vzxRd7ZZ1wzCuWd9vFe37mWeeUd++fUvs07BhQ7t5zurgdkbYAW4jzZo1kySlpKRI+r//s8/IyLDrV9KdVb+nZ8+eGjJkiA4dOqQlS5aoYsWK6ty58++u16pVK1WvXl1LlixRy5YttXHjRv31r3+16xMSEqJvvvlGhYWFdmd3fvjhB9vy6+Ht7a2ePXuqZ8+eys3NVbdu3fTWW28pLi7uqoOHixSdSSliGIaOHj1a7I/6s88+qy5dumjPnj1atGiR7rvvPt1zzz2/W9ejjz6qLVu26KefflKNGjV+91iKzr74+fkpMjLyd/tfj5CQEBUWFurYsWN2Z3MOHTpk16/oTq2CgoIy2zfgzLiMBTihTZs2lfh/60VjMor+kPn5+alKlSraunWrXb9Zs2bd8D6jo6Pl6uqq//znP1q6dKk6dep0Xc/gcXFxUffu3bVy5Ur961//Un5+vt0lLEnq0KGDUlNT7cb25Ofn67333pOPj881x8MUOXv2rN28h4eHwsPDZRiG8vLyfnf9hQsX6vz587b5ZcuWKSUlRe3bt7fr1759e1WpUkVvv/22tmzZ8rtndYqMHTtWBQUFeuaZZ0q8nPXb32fTpk1155136u9//3uJ/Uu6Bf73FB3L9OnT7dp/e3eVq6uroqOjtXz58hLPRpVm34Az48wO4ISGDRum7Oxsde3aVfXq1VNubq527NihJUuWqHbt2nbPtnn++ec1adIkPf/882rWrJm2bt2qw4cP3/A+q1Wrpscee0z/+Mc/dP78+WKB5Vp69uyp9957T+PGjVODBg1sY3GKDBo0SO+//7769eunxMRE1a5dW8uWLdP27ds1derUEp9L81tt27ZVUFCQHnroIQUGBur777/XjBkz1LFjx+taPyAgQC1btlT//v2VlpamqVOnKiwsTAMHDrTr5+7url69emnGjBlydXW1G1R9LQ8//LBmzJihYcOGqW7durYnKOfm5urw4cNatGiRPDw8FBQUJOlKSJw7d67at2+ve+65R/3791eNGjX0008/adOmTfLz89PKlSuva99FGjdurN69e2vWrFnKzMzUgw8+qA0bNujo0aPF+k6aNEmbNm1S8+bNNXDgQIWHh+vcuXPau3evvvjiC507d+6G9g04NcfdCAbgatasWWM899xzRr169QwfHx/Dw8PDCAsLM4YNG2akpaXZ9c3OzjYGDBhgWK1Ww9fX1+jRo4eRnp5+1VvPz5w5c9X9zpkzx5Bk+Pr6GpcuXSq2/Le3nhcpLCw0atasaUgy3nzzzRK3nZaWZvTv39+oUqWK4eHhYTRo0KDYLfNFt56/8847xdZ///33jVatWhmVK1c2PD09jTvvvNN4+eWXjczMzKsej2H8363T//nPf4y4uDijWrVqRoUKFYyOHTsaJ0+eLHGdotvu27Zte81tl+Trr782nn32WaNW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tpGUlKSOHTuqYsWKqlatml5++WXl5+ff6kMBAABOyuF3Y91zzz364osvbPNubv9X0ogRI7R69WotXbpUVqtVQ4cOVbdu3bR9+3ZJUkFBgTp27KigoCDt2LFDKSkpevbZZ+Xu7q4JEybc8mMBAADOx+Fhx83NrcQHv2VmZurDDz/U4sWL1bp1a0nS/PnzVb9+fe3cuVMtWrTQ559/ru+++05ffPGFAgMD1bhxY73xxhsaM2aMXnvtNXl4eNzqwwEAAE7G4WN2jhw5ouDgYNWpU0cxMTFKSkqSJCUmJiovL0+RkZG2vvXq1VOtWrWUkJAgSUpISFCDBg0UGBho6xMVFaWsrCwdPHjw1h4IAABwSg49s9O8eXPFx8fr7rvvVkpKisaPH6+HH35Y3377rVJTU+Xh4SF/f3+7dQIDA5WamipJSk1NtQs6RcuLll1NTk6OcnJybPNZWVlldEQAAMDZODTstG/f3vbvhg0bqnnz5goJCdHHH3+sChUqlNt+J06cqPHjx5fb9gEAgPNw+GWsX/P399ddd92lo0ePKigoSLm5ucrIyLDrk5aWZhvjExQUVOzurKL5a70AMi4uTpmZmbYpOTm5bA8EAAA4DacKOxcuXNCxY8dUvXp1NW3aVO7u7tqwYYNt+aFDh5SUlKSIiAhJUkREhA4cOKD09HRbn/Xr18vPz0/h4eFX3Y+np6f8/PzsJgAAYE4OvYw1atQode7cWSEhITp9+rTGjRsnV1dX9e7dW1arVQMGDNDIkSMVEBAgPz8/DRs2TBEREWrRooUkqW3btgoPD1efPn00efJkpaam6tVXX1VsbKw8PT0deWgAAMBJODTsnDp1Sr1799bZs2dVtWpVtWzZUjt37lTVqlUlSe+++65cXFwUHR2tnJwcRUVFadasWbb1XV1dtWrVKg0ePFgRERHy9vZW37599frrrzvqkAAAgJOxGIZhOLoIR8vKypLValVmZiaXtACUjsXRBQBOrJySxvX+/XaqMTsAAABljbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMjbADAABMzWnCzqRJk2SxWPTiiy/a2i5fvqzY2FhVrlxZPj4+io6OVlpamt16SUlJ6tixoypWrKhq1arp5ZdfVn5+/i2uHgAAOCunCDt79uzR+++/r4YNG9q1jxgxQitXrtTSpUu1ZcsWnT59Wt26dbMtLygoUMeOHZWbm6sdO3ZowYIFio+P19ixY2/1IQAAACfl8LBz4cIFxcTEaM6cOapUqZKtPTMzUx9++KH+8Y9/qHXr1mratKnmz5+vHTt2aOfOnZKkzz//XN99953+/e9/q3Hjxmrfvr3eeOMNzZw5U7m5uY46JAAA4EQcHnZiY2PVsWNHRUZG2rUnJiYqLy/Prr1evXqqVauWEhISJEkJCQlq0KCBAgMDbX2ioqKUlZWlgwcP3poDAAAATs3NkTv/6KOPtHfvXu3Zs6fYstTUVHl4eMjf39+uPTAwUKmpqbY+vw46RcuLll1NTk6OcnJybPNZWVmlPQQAAODkHHZmJzk5WS+88IIWLVokLy+vW7rviRMnymq12qaaNWve0v0DAIBbx2FhJzExUenp6WrSpInc3Nzk5uamLVu2aPr06XJzc1NgYKByc3OVkZFht15aWpqCgoIkSUFBQcXuziqaL+pTkri4OGVmZtqm5OTksj04AADgNBwWdh5//HEdOHBA+/bts03NmjVTTEyM7d/u7u7asGGDbZ1Dhw4pKSlJERERkqSIiAgdOHBA6enptj7r16+Xn5+fwsPDr7pvT09P+fn52U0AAMCcHDZmx9fXV/fee69dm7e3typXrmxrHzBggEaOHKmAgAD5+flp2LBhioiIUIsWLSRJbdu2VXh4uPr06aPJkycrNTVVr776qmJjY+Xp6XnLjwkAADgfhw5Q/j3vvvuuXFxcFB0drZycHEVFRWnWrFm25a6urlq1apUGDx6siIgIeXt7q2/fvnr99dcdWDUAAHAmFsMwDEcX4WhZWVmyWq3KzMzkkhaA0rE4ugDAiZVT0rjev98Of84O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p2vNmjWqX7++/Pz89PTTTys7O9u23tq1a9WyZUv5+/urcuXK6tSpk44dO3bN+r799lu1b99ePj4+CgwMVJ8+ffTzzz+X288D+CN69NFHNWzYML344ouqVKmSAgMDNWfOHNvT8X19fRUWFqY1a9ZIkgoKCjRgwADbd8Tdd9+tadOmXXMfv/e9AvMg7MDpTJs2Ta+//rruuOMOpaSkaM+ePZo4caIWLlyo2bNn6+DBgxoxYoSeeeYZbdmyxW7d1157TTNmzNCOHTuUnJysHj16aOrUqVq8eLFWr16tzz//XO+9956t/8WLFzVy5Eh99dVX2rBhg1xcXNS1a1cVFhaWWFtGRoZat26t++67T1999ZXWrl2rtLQ09ejRo1x/JsAf0YIFC1SlShXt3r1bw4YN0+DBg/XUU0/pwQcf1N69e9W2bVv16dNH2dnZKiws1B133KGlS5fqu+++09ixY/XKK6/o448/vur2r/d7BSZgAE7o3XffNUJCQgzDMIzLly8bFStWNHbs2GHXZ8CAAUbv3r0NwzCMTZs2GZKML774wrZ84sSJhiTj2LFjtrY//elPRlRU1FX3e+bMGUOSceDAAcMwDOP48eOGJOPrr782DMMw3njjDaNt27Z26yQnJxuSjEOHDpX6eAHYe+SRR4yWLVva5vPz8w1vb2+jT58+traUlBRDkpGQkFDiNmJjY43o6GjbfN++fY0uXboYhnF93yswD8bswOkdPXpU2dnZatOmjV17bm6u7rvvPru2hg0b2v4dGBioihUrqk6dOnZtu3fvts0fOXJEY8eO1a5du/Tzzz/bzugkJSXp3nvvLVbL/v37tWnTJvn4+BRbduzYMd11112lO0gAxfz68+zq6qrKlSurQYMGtrai9x+mp6dLkmbOnKl58+YpKSlJly5dUm5urho3blzitm/kewW3P8IOnN6FCxckSatXr1aNGjXslv323Tfu7u62f1ssFrv5orZfX6Lq3LmzQkJCNGfOHAUHB6uwsFD33nuvcnNzr1pL586d9fbbbxdbVr169Rs7MADXVNLn97efcenK2JuPPvpIo0aN0pQpUxQRESFfX1+988472rVrV4nbvpHvFdz+CDtweuHh4fL09FRSUpIeeeSRMtvu2bNndejQIc2ZM0cPP/ywJOnLL7+85jpNmjTR8uXLVbt2bbm58fEBnMX27dv14IMPasiQIba2a91sUF7fK3BOfFvD6fn6+mrUqFEaMWKECgsL1bJlS2VmZmr79u3y8/NT3759S7XdSpUqqXLlyvrggw9UvXp1JSUl6S9/+cs114mNjdWcOXPUu3dvjR49WgEBATp69Kg++ugjzZ07V66urqWqBcDNqVu3rhYuXKh169YpNDRU//rXv7Rnzx6FhoaW2L+8vlfgnAg7uC288cYbqlq1qiZOnKgff/xR/v7+atKkiV555ZVSb9PFxUUfffSRhg8frnvvvVd33323pk+frkcfffSq6wQHB2v79u0aM2aM2rZtq5ycHIWEhKhdu3ZyceHmRsBR/vSnP+nrr79Wz549ZbFY1Lt3bw0ZMsR2a3pJyuN7Bc7JYhiG4egiAAAAygv/KwoAAEyNsAMAAEyNsAMAAEyNsAMAAEyNsAMAAEyNsAMAAEyNsAMAAEyNsAMAAEyNsAPgtnTmzBkNHjxYtWrVkqenp4KCghQVFaXt27c7ujQATobXRQC4LUVHRys3N1cLFixQnTp1lJaWpg0bNujs2bOOLg2Ak+HMDoDbTkZGhrZt26a3335bjz32mEJCQvTAAw8oLi5OTzzxhK3P888/r6pVq8rPz0+tW7fW/v37JV05KxQUFKQJEybYtrljxw55eHhow4YNDjkmAOWHsAPgtuPj4yMfHx+tWLFCOTk5JfZ56qmnlJ6erjVr1igxMVFNmjTR448/rnPnzqlq1aqaN2+eXnvtNX311Vc6f/68+vTpo6FDh+rxxx+/xUcDoLzxIlAAt6Xly5dr4MCBunTpkpo0aaJHHnlEvXr1UsOGDfXll1+qY8eOSk9Pl6enp22dsLAwjR49WoMGDZIkxcbG6osvvlCzZs104MAB7dmzx64/AHMg7AC4bV2+fFnbtm3Tzp07tWbNGu3evVtz587VxYsXNXz4cFWoUMGu/6VLlzRq1Ci9/fbbtvl7771XycnJSkxMVIMGDRxxGADKGWEHgGk8//zzWr9+vYYMGaL33ntPmzdvLtbH399fVapUkSR9++23uv/++5WXl6dPP/1UnTt3vsUVA7gVuBsLgGmEh4drxYoVatKkiVJTU+Xm5qbatWuX2Dc3N1fPPPOMevbsqbvvvlvPP/+8Dhw4oGrVqt3aogGUO87sALjtnD17Vk899ZSee+45NWzYUL6+vvrqq680bNgwdezYUXPnzlWrVq10/vx5TZ48WXfddZdOnz6t1atXq2vXrmrWrJlefvllLVu2TPv375ePj48eeeQRWa1WrVq1ytGHB6CMEXYA3HZycnL02muv6fPPP9exY8eUl5enmjVr6qmnntIrr7yiChUq6Pz58/rrX/+q5cuX2241b9WqlSZOnKhjx46pTZs22rRpk1q2bClJOnHihBo1aqRJkyZp8ODBDj5CAGWJsAMAAEyN5+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABTI+wAAABT+3/JYyXEt1NNgAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "x = range(len(survivetotalsex.index.values))\n",
    "ht = survivetotalsex.Total\n",
    "hs = survivetotalsex.Survived\n",
    "\n",
    "pht = pl.bar(x, ht,color=['magenta'])\n",
    "phs = pl.bar(x, hs, color=['cyan'])\n",
    "\n",
    "pl.xticks(x, survivetotalsex.index.values)\n",
    "pl.xlabel('Sex')\n",
    "pl.ylabel('Passengers')\n",
    "pl.title('Survivors by Gender')\n",
    "\n",
    "pl.legend([pht,phs],['Died', 'Survived'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Determine the survival rate is associated to the age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "177"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanicdf['Age'].isnull().sum()\n",
    "#Out of 891 passengers, there is 177 missing Age values, so the result might be less reliable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>20-29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>30-39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>20-29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>30-39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>30-39</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked AgeGroup  \n",
       "0      0         A/5 21171   7.2500   NaN        S    20-29  \n",
       "1      0          PC 17599  71.2833   C85        C    30-39  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S    20-29  \n",
       "3      0            113803  53.1000  C123        S    30-39  \n",
       "4      0            373450   8.0500   NaN        S    30-39  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def age_group(age):\n",
    "    if age >= 80:\n",
    "        return '80-89'\n",
    "    if age >= 70:\n",
    "        return '70-79'\n",
    "    if age >= 60:\n",
    "        return '60-69'\n",
    "    if age >= 50:\n",
    "        return '50-59'\n",
    "    if age >= 40:\n",
    "        return '40-49'\n",
    "    if age >= 30:\n",
    "        return '30-39'\n",
    "    if age >= 20:\n",
    "        return '20-29'\n",
    "    if age >= 10:\n",
    "        return '10-19'\n",
    "    if age >= 0:\n",
    "        return '0-9'\n",
    "    \n",
    "titanicdf['AgeGroup'] = titanicdf.Age.apply(age_group)\n",
    "titanicdf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AgeGroup</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0-9</th>\n",
       "      <td>38</td>\n",
       "      <td>62</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10-19</th>\n",
       "      <td>41</td>\n",
       "      <td>102</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20-29</th>\n",
       "      <td>77</td>\n",
       "      <td>220</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30-39</th>\n",
       "      <td>73</td>\n",
       "      <td>167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40-49</th>\n",
       "      <td>34</td>\n",
       "      <td>89</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50-59</th>\n",
       "      <td>20</td>\n",
       "      <td>48</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60-69</th>\n",
       "      <td>6</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70-79</th>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>80-89</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Survived  Total\n",
       "AgeGroup                 \n",
       "0-9             38     62\n",
       "10-19           41    102\n",
       "20-29           77    220\n",
       "30-39           73    167\n",
       "40-49           34     89\n",
       "50-59           20     48\n",
       "60-69            6     19\n",
       "70-79            0      6\n",
       "80-89            1      1"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agesummary = titanicdf.groupby(['AgeGroup'], as_index=False)['Survived'].agg([np.sum, np.size])\n",
    "agesummary = agesummary.rename(columns={'sum':'Survived', 'size':'Total'})\n",
    "agesummary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "      <th>Percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>233</td>\n",
       "      <td>314</td>\n",
       "      <td>74.203822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>109</td>\n",
       "      <td>577</td>\n",
       "      <td>18.890815</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Survived  Total  Percentage\n",
       "Sex                                \n",
       "female       233    314   74.203822\n",
       "male         109    577   18.890815"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "percent_survived = (survivetotalsex['Survived'] / survivetotalsex['Total']) * 100\n",
    "survivetotalsex['Percentage'] = percent_survived\n",
    "\n",
    "survivetotalsex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1c09a9e4cd0>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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afr7u0zN2SEiIiIqKoYdIiIikjWGHSIiIpI1hh0iIiKSNYYdIiIikjWGHSIiIpI1hh0iIiKSNYYdIiIikrVSfQdlKl0UZeDmY2WgRCIiest4ZoeIiIhkjWGHiIiIZI1hh4iIiGSNYYeIiIhkjROU9YyTeomIiEoWz+wQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkazxaiwiKlV4BSMR6RrP7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsMewQERGRrDHsEBERkawx7BAREZGsGZV0AUREcqUQJV3Bm5WBEomKjWGHiIg0oyjpAjTA9EYF4MdYREREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrDDtEREQkaww7REREJGsMO0RERCRrRiVdABERlQ0KUdIVvJnGJSr0WYWOlIGfd1nBMztEREQkaww7REREJGsMO0RERCRrDDtEREQka7IJO4sWLULVqlVhamoKDw8PnDlzpqRLIiIiolJAFmFn48aNmDBhAoKDg3Hu3Dk0atQI3t7eSE5OLunSiIiIqITJIuz88ssvGDp0KAYPHoy6deti6dKlMDc3x8qVK0u6NCIiIiphZT7sZGdnIyoqCl5eXtIyAwMDeHl54eTJkyVYGRERkf4pysCjpJX5mwr+73//Q15eHhwdHdWWOzo64vr16wW+JisrC1lZWdLz1NRUAEBaWpr+Ci3F5NS1nHoB5NUPeymd5NQLILN+ZNSMvlp5+XtbiNffgbHMhx1tzJw5E9OmTcu3vEqVKiVQTcmzKekCdEhOvQDy6oe9lE5y6gWQWT8yakbfraSnp8PGpvC9lPmwU758eRgaGiIpKUlteVJSEpycnAp8TVBQECZMmCA9V6lUePz4McqVKweFojSccCtcWloaqlSpgvj4eFhbW5d0OcXCXkonOfUCyKsf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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "x = range(len(agesummary.index.values))\n",
    "ht = agesummary.Total\n",
    "hs = agesummary.Survived\n",
    "\n",
    "pht = pl.bar(x, ht, color=['magenta'])\n",
    "phs = pl.bar(x, hs, color=['cyan'])\n",
    "\n",
    "pl.xticks(x, agesummary.index.values)\n",
    "pl.xlabel('Age groups')\n",
    "pl.ylabel('Passengers')\n",
    "pl.title('Survivors by Age group')\n",
    "\n",
    "\n",
    "pl.legend([pht,phs],['Died', 'Survived'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Total</th>\n",
       "      <th>SurvivedPercent</th>\n",
       "      <th>DiedPercent</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AgeGroup</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0-9</th>\n",
       "      <td>38</td>\n",
       "      <td>62</td>\n",
       "      <td>61.29</td>\n",
       "      <td>38.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10-19</th>\n",
       "      <td>41</td>\n",
       "      <td>102</td>\n",
       "      <td>40.20</td>\n",
       "      <td>59.80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20-29</th>\n",
       "      <td>77</td>\n",
       "      <td>220</td>\n",
       "      <td>35.00</td>\n",
       "      <td>65.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30-39</th>\n",
       "      <td>73</td>\n",
       "      <td>167</td>\n",
       "      <td>43.71</td>\n",
       "      <td>56.29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40-49</th>\n",
       "      <td>34</td>\n",
       "      <td>89</td>\n",
       "      <td>38.20</td>\n",
       "      <td>61.80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50-59</th>\n",
       "      <td>20</td>\n",
       "      <td>48</td>\n",
       "      <td>41.67</td>\n",
       "      <td>58.33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60-69</th>\n",
       "      <td>6</td>\n",
       "      <td>19</td>\n",
       "      <td>31.58</td>\n",
       "      <td>68.42</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70-79</th>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>0.00</td>\n",
       "      <td>100.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>80-89</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>100.00</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Survived  Total  SurvivedPercent  DiedPercent\n",
       "AgeGroup                                               \n",
       "0-9             38     62            61.29        38.71\n",
       "10-19           41    102            40.20        59.80\n",
       "20-29           77    220            35.00        65.00\n",
       "30-39           73    167            43.71        56.29\n",
       "40-49           34     89            38.20        61.80\n",
       "50-59           20     48            41.67        58.33\n",
       "60-69            6     19            31.58        68.42\n",
       "70-79            0      6             0.00       100.00\n",
       "80-89            1      1           100.00         0.00"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agesummary['SurvivedPercent'] = round((agesummary.Survived / agesummary.Total) * 100,2)\n",
    "agesummary['DiedPercent'] = round(((agesummary.Total - agesummary.Survived) / agesummary.Total) * 100,2)\n",
    "agesummary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above graph and percentages, shockingly we can see that most who died were from 20-29 age group, when this is the age group that can has the capability to save themselves.\n",
    "\n",
    "Also, survival percentage of 0-9 age group is best - at 61.29% which means to say that children are more prioritized during rescue operations."
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
