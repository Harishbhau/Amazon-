{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Classification with Natural Language Processing on LSTM "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing the libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Reviews.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df[:10000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
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
       "      <th>ProductId</th>\n",
       "      <th>UserId</th>\n",
       "      <th>ProfileName</th>\n",
       "      <th>HelpfulnessNumerator</th>\n",
       "      <th>HelpfulnessDenominator</th>\n",
       "      <th>Score</th>\n",
       "      <th>Time</th>\n",
       "      <th>Summary</th>\n",
       "      <th>Text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>B001E4KFG0</td>\n",
       "      <td>A3SGXH7AUHU8GW</td>\n",
       "      <td>delmartian</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1303862400</td>\n",
       "      <td>Good Quality Dog Food</td>\n",
       "      <td>I have bought several of the Vitality canned d...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>B00813GRG4</td>\n",
       "      <td>A1D87F6ZCVE5NK</td>\n",
       "      <td>dll pa</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1346976000</td>\n",
       "      <td>Not as Advertised</td>\n",
       "      <td>Product arrived labeled as Jumbo Salted Peanut...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>B000LQOCH0</td>\n",
       "      <td>ABXLMWJIXXAIN</td>\n",
       "      <td>Natalia Corres \"Natalia Corres\"</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1219017600</td>\n",
       "      <td>\"Delight\" says it all</td>\n",
       "      <td>This is a confection that has been around a fe...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>B000UA0QIQ</td>\n",
       "      <td>A395BORC6FGVXV</td>\n",
       "      <td>Karl</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1307923200</td>\n",
       "      <td>Cough Medicine</td>\n",
       "      <td>If you are looking for the secret ingredient i...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>B006K2ZZ7K</td>\n",
       "      <td>A1UQRSCLF8GW1T</td>\n",
       "      <td>Michael D. Bigham \"M. Wassir\"</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>1350777600</td>\n",
       "      <td>Great taffy</td>\n",
       "      <td>Great taffy at a great price.  There was a wid...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id   ProductId          UserId                      ProfileName  \\\n",
       "0   1  B001E4KFG0  A3SGXH7AUHU8GW                       delmartian   \n",
       "1   2  B00813GRG4  A1D87F6ZCVE5NK                           dll pa   \n",
       "2   3  B000LQOCH0   ABXLMWJIXXAIN  Natalia Corres \"Natalia Corres\"   \n",
       "3   4  B000UA0QIQ  A395BORC6FGVXV                             Karl   \n",
       "4   5  B006K2ZZ7K  A1UQRSCLF8GW1T    Michael D. Bigham \"M. Wassir\"   \n",
       "\n",
       "   HelpfulnessNumerator  HelpfulnessDenominator  Score        Time  \\\n",
       "0                     1                       1      5  1303862400   \n",
       "1                     0                       0      1  1346976000   \n",
       "2                     1                       1      4  1219017600   \n",
       "3                     3                       3      2  1307923200   \n",
       "4                     0                       0      5  1350777600   \n",
       "\n",
       "                 Summary                                               Text  \n",
       "0  Good Quality Dog Food  I have bought several of the Vitality canned d...  \n",
       "1      Not as Advertised  Product arrived labeled as Jumbo Salted Peanut...  \n",
       "2  \"Delight\" says it all  This is a confection that has been around a fe...  \n",
       "3         Cough Medicine  If you are looking for the secret ingredient i...  \n",
       "4            Great taffy  Great taffy at a great price.  There was a wid...  "
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 10)"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# Text Cleaning or Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to C:\\Users\\Aniruddha\n",
      "[nltk_data]     choudhury\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# Cleaning the texts\n",
    "import re\n",
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "corpus = []\n",
    "for i in range(0, 10000):\n",
    "    review = re.sub('[^a-zA-Z]', ' ', df['Text'][i])\n",
    "    review = review.lower()\n",
    "    review = review.split()\n",
    "    ps = PorterStemmer()\n",
    "    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]\n",
    "    review = ' '.join(review)\n",
    "    corpus.append(review)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
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
       "      <th>Reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>bought sever vital can dog food product found ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>product arriv label jumbo salt peanut peanut a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>confect around centuri light pillowi citru gel...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>look secret ingredi robitussin believ found go...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>great taffi great price wide assort yummi taff...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             Reviews\n",
       "0  bought sever vital can dog food product found ...\n",
       "1  product arriv label jumbo salt peanut peanut a...\n",
       "2  confect around centuri light pillowi citru gel...\n",
       "3  look secret ingredi robitussin believ found go...\n",
       "4  great taffi great price wide assort yummi taff..."
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corpus=pd.DataFrame(corpus, columns=['Reviews']) \n",
    "corpus.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
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
       "      <th>Reviews</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>bought sever vital can dog food product found ...</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>product arriv label jumbo salt peanut peanut a...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>confect around centuri light pillowi citru gel...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>look secret ingredi robitussin believ found go...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>great taffi great price wide assort yummi taff...</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             Reviews  Score\n",
       "0  bought sever vital can dog food product found ...      5\n",
       "1  product arriv label jumbo salt peanut peanut a...      1\n",
       "2  confect around centuri light pillowi citru gel...      4\n",
       "3  look secret ingredi robitussin believ found go...      2\n",
       "4  great taffi great price wide assort yummi taff...      5"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result=corpus.join(df[['Score']])\n",
    "result.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TFIDF"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "TFIDF is an information retrieval technique that weighs a term’s frequency (TF) and its inverse document frequency (IDF). Each word has its respective TF and IDF score. The product of the TF and IDF scores of a word is called the TFIDF weight of that word.\n",
    "\n",
    "Put simply, the higher the TFIDF score (weight), the rarer the word and vice versa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "        dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',\n",
       "        lowercase=True, max_df=1.0, max_features=None, min_df=1,\n",
       "        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,\n",
       "        stop_words=None, strip_accents=None, sublinear_tf=False,\n",
       "        token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', tokenizer=None, use_idf=True,\n",
       "        vocabulary=None)"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "tfidf = TfidfVectorizer()\n",
    "tfidf.fit(result['Reviews'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'product arriv label jumbo salt peanut peanut actual small size unsalt sure error vendor intend repres product jumbo'"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = tfidf.transform(result['Reviews'])\n",
    "result['Reviews'][1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.37509695630835344]\n"
     ]
    }
   ],
   "source": [
    "print([X[1, tfidf.vocabulary_['peanut']]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.5727828521365451]\n"
     ]
    }
   ],
   "source": [
    "print([X[1, tfidf.vocabulary_['jumbo']]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.2706003734708304]\n"
     ]
    }
   ],
   "source": [
    "print([X[1, tfidf.vocabulary_['error']]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Among the three words, “peanut”, “jumbo” and “error”, tf-idf gives the highest weight to “jumbo”. Why? This indicates that “jumbo” is a much rarer word than “peanut” and “error”. This is how to use the tf-idf to indicate the importance of words or terms inside a collection of documents."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
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
       "      <th>Reviews</th>\n",
       "      <th>Positivity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>bought sever vital can dog food product found ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>product arriv label jumbo salt peanut peanut a...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>confect around centuri light pillowi citru gel...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>look secret ingredi robitussin believ found go...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>great taffi great price wide assort yummi taff...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             Reviews  Positivity\n",
       "0  bought sever vital can dog food product found ...           1\n",
       "1  product arriv label jumbo salt peanut peanut a...           0\n",
       "2  confect around centuri light pillowi citru gel...           1\n",
       "3  look secret ingredi robitussin believ found go...           0\n",
       "4  great taffi great price wide assort yummi taff...           1"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.dropna(inplace=True)\n",
    "result[result['Score'] != 3]\n",
    "result['Positivity'] = np.where(result['Score'] > 3, 1, 0)\n",
    "cols = [ 'Score']\n",
    "result.drop(cols, axis=1, inplace=True)\n",
    "result.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Positivity\n",
       "0    2384\n",
       "1    7616\n",
       "dtype: int64"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.groupby('Positivity').size()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X = result.Reviews\n",
    "y = result.Positivity\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set has total 7500 entries with 23.57% negative, 76.43% positive\n"
     ]
    }
   ],
   "source": [
    "print(\"Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive\".format(len(X_train),\n",
    "                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,\n",
    "                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))"
   ]
  
