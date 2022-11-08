{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "## SF Salaries Exercise \n",
    "\n",
    "\n",
    "For this exercise, you will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle!. You can also used the SF Salaries Dataset on D2L to answer the questions. Please download the \"Salaries.csv\" file from kaggle or D2L and complete the below exercises. Please submit the completed jupyter notebook file to D2L. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 1**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Import pandas as pd.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Read Salaries.csv as a dataframe called sal.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "sal = pd.read_csv(\"Salaries.csv\") "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Check the head of the DataFrame.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "      <th>Id</th>\n",
       "      <th>EmployeeName</th>\n",
       "      <th>JobTitle</th>\n",
       "      <th>BasePay</th>\n",
       "      <th>OvertimePay</th>\n",
       "      <th>OtherPay</th>\n",
       "      <th>Benefits</th>\n",
       "      <th>TotalPay</th>\n",
       "      <th>TotalPayBenefits</th>\n",
       "      <th>Year</th>\n",
       "      <th>Notes</th>\n",
       "      <th>Agency</th>\n",
       "      <th>Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>NATHANIEL FORD</td>\n",
       "      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>\n",
       "      <td>167411.18</td>\n",
       "      <td>0.00</td>\n",
       "      <td>400184.25</td>\n",
       "      <td>NaN</td>\n",
       "      <td>567595.43</td>\n",
       "      <td>567595.43</td>\n",
       "      <td>2011</td>\n",
       "      <td>NaN</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>GARY JIMENEZ</td>\n",
       "      <td>CAPTAIN III (POLICE DEPARTMENT)</td>\n",
       "      <td>155966.02</td>\n",
       "      <td>245131.88</td>\n",
       "      <td>137811.38</td>\n",
       "      <td>NaN</td>\n",
       "      <td>538909.28</td>\n",
       "      <td>538909.28</td>\n",
       "      <td>2011</td>\n",
       "      <td>NaN</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>ALBERT PARDINI</td>\n",
       "      <td>CAPTAIN III (POLICE DEPARTMENT)</td>\n",
       "      <td>212739.13</td>\n",
       "      <td>106088.18</td>\n",
       "      <td>16452.60</td>\n",
       "      <td>NaN</td>\n",
       "      <td>335279.91</td>\n",
       "      <td>335279.91</td>\n",
       "      <td>2011</td>\n",
       "      <td>NaN</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>CHRISTOPHER CHONG</td>\n",
       "      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>\n",
       "      <td>77916.00</td>\n",
       "      <td>56120.71</td>\n",
       "      <td>198306.90</td>\n",
       "      <td>NaN</td>\n",
       "      <td>332343.61</td>\n",
       "      <td>332343.61</td>\n",
       "      <td>2011</td>\n",
       "      <td>NaN</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>PATRICK GARDNER</td>\n",
       "      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>\n",
       "      <td>134401.60</td>\n",
       "      <td>9737.00</td>\n",
       "      <td>182234.59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>326373.19</td>\n",
       "      <td>326373.19</td>\n",
       "      <td>2011</td>\n",
       "      <td>NaN</td>\n",
       "      <td>San Francisco</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id       EmployeeName                                        JobTitle  \\\n",
       "0   1     NATHANIEL FORD  GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY   \n",
       "1   2       GARY JIMENEZ                 CAPTAIN III (POLICE DEPARTMENT)   \n",
       "2   3     ALBERT PARDINI                 CAPTAIN III (POLICE DEPARTMENT)   \n",
       "3   4  CHRISTOPHER CHONG            WIRE ROPE CABLE MAINTENANCE MECHANIC   \n",
       "4   5    PATRICK GARDNER    DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)   \n",
       "\n",
       "     BasePay  OvertimePay   OtherPay  Benefits   TotalPay  TotalPayBenefits  \\\n",
       "0  167411.18         0.00  400184.25       NaN  567595.43         567595.43   \n",
       "1  155966.02    245131.88  137811.38       NaN  538909.28         538909.28   \n",
       "2  212739.13    106088.18   16452.60       NaN  335279.91         335279.91   \n",
       "3   77916.00     56120.71  198306.90       NaN  332343.61         332343.61   \n",
       "4  134401.60      9737.00  182234.59       NaN  326373.19         326373.19   \n",
       "\n",
       "   Year  Notes         Agency  Status  \n",
       "0  2011    NaN  San Francisco     NaN  \n",
       "1  2011    NaN  San Francisco     NaN  \n",
       "2  2011    NaN  San Francisco     NaN  \n",
       "3  2011    NaN  San Francisco     NaN  \n",
       "4  2011    NaN  San Francisco     NaN  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 2 - Use the .info() method to find out how many entries there are.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 148654 entries, 0 to 148653\n",
      "Data columns (total 13 columns):\n",
      " #   Column            Non-Null Count   Dtype  \n",
      "---  ------            --------------   -----  \n",
      " 0   Id                148654 non-null  int64  \n",
      " 1   EmployeeName      148654 non-null  object \n",
      " 2   JobTitle          148654 non-null  object \n",
      " 3   BasePay           148045 non-null  float64\n",
      " 4   OvertimePay       148650 non-null  float64\n",
      " 5   OtherPay          148650 non-null  float64\n",
      " 6   Benefits          112491 non-null  float64\n",
      " 7   TotalPay          148654 non-null  float64\n",
      " 8   TotalPayBenefits  148654 non-null  float64\n",
      " 9   Year              148654 non-null  int64  \n",
      " 10  Notes             0 non-null       float64\n",
      " 11  Agency            148654 non-null  object \n",
      " 12  Status            0 non-null       float64\n",
      "dtypes: float64(8), int64(2), object(3)\n",
      "memory usage: 14.7+ MB\n"
     ]
    }
   ],
   "source": [
    "sal.info()\n",
    "# there are 148,654 entries"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 3 - What is the average BasePay ?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "66325.45"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "round(sal['BasePay'].mean(),2)    # the Average for BasePay is 66,325.45"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 4 - What is the highest amount of OvertimePay in the dataset ?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "245131.88"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal['OvertimePay'].max()   # the OvertimePay max value is 245,131.88"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 5 - What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll).**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "24    CAPTAIN, FIRE SUPPRESSION\n",
       "Name: JobTitle, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[sal[\"EmployeeName\"] == \"JOSEPH DRISCOLL\"][\"JobTitle\"]\n",
    "# Joseph Driscoll's job title is CAPTAIN, FIRE SUPPRESSION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 6 - How much does JOSEPH DRISCOLL make (including benefits)?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "24    270324.91\n",
       "Name: TotalPayBenefits, dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[sal[\"EmployeeName\"] == \"JOSEPH DRISCOLL\"][\"TotalPayBenefits\"]\n",
    "# Joseph Driscoll is earning 270,324.91 including benefits"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 7 - What is the name of highest paid person (including benefits)?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    NATHANIEL FORD\n",
       "Name: EmployeeName, dtype: object"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[sal[\"TotalPayBenefits\"] == sal[\"TotalPayBenefits\"].max()][\"EmployeeName\"]\n",
    "# The highest paid person including benefits is NATHANIEL FORD"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 8 - What is the name of lowest paid person (including benefits)?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "148653    Joe Lopez\n",
       "Name: EmployeeName, dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[sal[\"TotalPayBenefits\"] == sal[\"TotalPayBenefits\"].min()][\"EmployeeName\"]\n",
    "# The lowest paid person is Joe Lopez"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 9 - What was the average (mean) BasePay of all employees per year? (2011-2014)?**"
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
       "Year\n",
       "2011    63595.956517\n",
       "2012    65436.406857\n",
       "2013    69630.030216\n",
       "2014    66564.421924\n",
       "Name: BasePay, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal.groupby('Year')[\"BasePay\"].mean() # computing the (mean) BasePay per year"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 10 - How many unique job titles are there?**"
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
       "2159"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[\"JobTitle\"].nunique()    # there are 2159 unique job titles"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 11 - What are the top 5 most common jobs?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Transit Operator                7036\n",
       "Special Nurse                   4389\n",
       "Registered Nurse                3736\n",
       "Public Svc Aide-Public Works    2518\n",
       "Police Officer 3                2421\n",
       "Name: JobTitle, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[\"JobTitle\"].value_counts().head()\n",
    "# The five most common jobs are Transit Operator, Special Nurse, Registered Nurse, Public Svc Aide-Public Works,Police Office 3 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 12 - How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id                  37606\n",
       "EmployeeName        36150\n",
       "JobTitle             1051\n",
       "BasePay             26979\n",
       "OvertimePay         16833\n",
       "OtherPay            21868\n",
       "Benefits            31388\n",
       "TotalPay            34998\n",
       "TotalPayBenefits    35706\n",
       "Year                    1\n",
       "Notes                   0\n",
       "Agency                  1\n",
       "Status                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jobs = sal[sal[\"Year\"] == 2013]\n",
    "# There were 1051 JobTitle recorded in 2013"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "202"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(jobs[\"JobTitle\"].value_counts() == 1)   # The occurence of only 1 person per job in 2013 is 202"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise 13: Is there a correlation between length of the Job Title string and Salary?**"
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
       "      <th>JobTitle</th>\n",
       "      <th>TotalPayBenefits</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>\n",
       "      <td>567595.43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CAPTAIN III (POLICE DEPARTMENT)</td>\n",
       "      <td>538909.28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>CAPTAIN III (POLICE DEPARTMENT)</td>\n",
       "      <td>335279.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>\n",
       "      <td>332343.61</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>\n",
       "      <td>326373.19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148649</th>\n",
       "      <td>Custodian</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148650</th>\n",
       "      <td>Not provided</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148651</th>\n",
       "      <td>Not provided</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148652</th>\n",
       "      <td>Not provided</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148653</th>\n",
       "      <td>Counselor, Log Cabin Ranch</td>\n",
       "      <td>-618.13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>148654 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              JobTitle  TotalPayBenefits\n",
       "0       GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY         567595.43\n",
       "1                      CAPTAIN III (POLICE DEPARTMENT)         538909.28\n",
       "2                      CAPTAIN III (POLICE DEPARTMENT)         335279.91\n",
       "3                 WIRE ROPE CABLE MAINTENANCE MECHANIC         332343.61\n",
       "4         DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)         326373.19\n",
       "...                                                ...               ...\n",
       "148649                                       Custodian              0.00\n",
       "148650                                    Not provided              0.00\n",
       "148651                                    Not provided              0.00\n",
       "148652                                    Not provided              0.00\n",
       "148653                      Counselor, Log Cabin Ranch           -618.13\n",
       "\n",
       "[148654 rows x 2 columns]"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[[\"JobTitle\",\"TotalPayBenefits\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "sal[\"titlelen\"] = sal[\"JobTitle\"].apply(len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
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
       "      <th>titlelen</th>\n",
       "      <th>TotalPayBenefits</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>46</td>\n",
       "      <td>567595.43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>31</td>\n",
       "      <td>538909.28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>31</td>\n",
       "      <td>335279.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>36</td>\n",
       "      <td>332343.61</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>44</td>\n",
       "      <td>326373.19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148649</th>\n",
       "      <td>9</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148650</th>\n",
       "      <td>12</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148651</th>\n",
       "      <td>12</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148652</th>\n",
       "      <td>12</td>\n",
       "      <td>0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148653</th>\n",
       "      <td>26</td>\n",
       "      <td>-618.13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>148654 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        titlelen  TotalPayBenefits\n",
       "0             46         567595.43\n",
       "1             31         538909.28\n",
       "2             31         335279.91\n",
       "3             36         332343.61\n",
       "4             44         326373.19\n",
       "...          ...               ...\n",
       "148649         9              0.00\n",
       "148650        12              0.00\n",
       "148651        12              0.00\n",
       "148652        12              0.00\n",
       "148653        26           -618.13\n",
       "\n",
       "[148654 rows x 2 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[[\"titlelen\",\"TotalPayBenefits\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
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
       "      <th>titlelen</th>\n",
       "      <th>TotalPayBenefits</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>titlelen</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.036878</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TotalPayBenefits</th>\n",
       "      <td>-0.036878</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  titlelen  TotalPayBenefits\n",
       "titlelen          1.000000         -0.036878\n",
       "TotalPayBenefits -0.036878          1.000000"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sal[[\"titlelen\", \"TotalPayBenefits\"]].corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# The correlation of -0.036878 is negligible between Job Title length and salary."
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
