
import json
from collections import Counter
import numpy as np
import pandas as pd
import re
import nltk
from nltk.data import find
import gensim
import sklearn
from sympy.parsing.sympy_parser import parse_expr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

np.random.seed(0)
nltk.download('word2vec_sample')


class NgramLM:
    
    def __init__(self):

        # Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram = {}
        
        # Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram_weights = {}

    def load_trigrams(self):
        """
        Loads the trigrams from the data file and fills the dictionaries defined above.

        """
        with open("data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
            lines = f.readlines()
            for line in lines:
                word1, word2, word3, count = line.strip().split()
                if (word1, word2) not in self.bigram_prefix_to_trigram:
                    self.bigram_prefix_to_trigram[(word1, word2)] = []
                    self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
                self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
                self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

    def top_next_word(self, word1, word2, n=10):
        
        next_words = []
        probs = []
        bigram = (word1, word2)
        
        if bigram not in self.bigram_prefix_to_trigram:
            return [], []

        words = self.bigram_prefix_to_trigram[bigram]
        counts = self.bigram_prefix_to_trigram_weights[bigram]

        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        top_indices = sorted_indices[:n]
        
        next_words = [words[i] for i in top_indices]
        total_count = sum(counts)
        probs = [counts[i] / total_count for i in top_indices]

        return next_words, probs

    def sample_next_word(self, word1, word2, n=10):
        """
        Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

        """
        next_words = []
        probs = []
        bigram = (word1, word2)
        
        if bigram not in self.bigram_prefix_to_trigram:
            return [], [] 
        
        words = self.bigram_prefix_to_trigram[bigram]
        counts = self.bigram_prefix_to_trigram_weights[bigram]
        
        total_count = sum(counts)
        probabilities = [count / total_count for count in counts]
        indices = np.random.choice(len(words), size=min(n, len(words)), replace=False, p=probabilities)
        next_words = [words[i] for i in indices]
        
        probs = [probabilities[i] for i in indices]
        
        return next_words, probs


    def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
        """
        Generate sentences using beam search.

        """
        sentences = []
        probs = []
        beam_queue = [(prefix.split(), 1.0)]
        
        for _ in range(max_len - len(prefix.split())):
            new_beam_queue = []
            
            for sentence, prob in beam_queue:
                if sentence[-1] == "<EOS>":  
                    new_beam_queue.append((sentence, prob))
                    continue  
                
                if len(sentence) < 2:
                    continue  
                word1, word2 = sentence[-2], sentence[-1]
                next_words, next_probs = sampler(word1, word2, n=beam)  
                
                for next_word, next_prob in zip(next_words, next_probs):
                    new_sentence = sentence + [next_word]
                    new_prob = prob * next_prob  
                    new_beam_queue.append((new_sentence, new_prob))
            
            if not new_beam_queue:
                break  
            
            beam_queue = sorted(new_beam_queue, key=lambda x: x[1], reverse=True)[:beam]
        
        for sentence, prob in beam_queue:
            if sentence[-1] != "<EOS>":
                sentence.append("<EOS>")  
            sentences.append(" ".join(sentence))
            probs.append(prob)
        
        return sentences, probs


print("======================================================================")
print("Checking Language Model")
print("======================================================================")

# Defines language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()

print("------------- Evaluating top next word prediction -------------")
next_words, probs = language_model.top_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)

print("------------- Evaluating sample next word prediction -------------")
next_words, probs = language_model.sample_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)

print("------------- Evaluating beam search -------------")
sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")


sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
    print(sent, prob)



class Text2SQLParser:
    def __init__(self):
        """
        Basic Text2SQL Parser. This module just attempts to classify the user queries into different "categories" of SQL queries.
        """
        self.parser_files = "data/semantic-parser"
        self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

        self.train_file = "sql_train.tsv"
        self.test_file = "sql_val.tsv"

    def load_data(self):
        """
        Load the data from file.

        """
        self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
        self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")

        self.ls_labels = list(self.train_df["Label"].unique())

    def predict_label_using_keywords(self, question):
        """
        Predicts the label for the question using custom-defined keywords.

        """

        label= ""
        keywords = {

            'comparison': {'greater', 'less', 'equal', 'between', 'compare', 'difference', 'versus', 'match'},
            'grouping': {'group', 'count', 'sum', 'average', 'having', 'aggregate', 'total', 'each', 'correspond'},
            'ordering': {'order', 'sort', 'rank', 'top', 'limit', 'ascending', 'descending'},
            'multi_table': {'join', 'foreign key', 'inner', 'outer', 'left', 'right', 'cross', 'combine', 'dataset', 'merge', 'relation', 'physician', 'order_id', 'address_id', 'product_id', 'customer_id', 'patient', 'dept_code', 'other_details'}
        }
        

        question_lower = question.lower()
        
        multi_table_matches = [word for word in keywords['multi_table'] if word in question_lower]
        if len(multi_table_matches) >= 2:
            return 'multi_table'
        match_counts = {label: sum(word in question_lower for word in words) for label, words in keywords.items()}

        label = max(match_counts, key=match_counts.get, default='comparison')

            
        return label
    
    def evaluate_accuracy(self, prediction_function_name):
        """
        Gives label wise accuracy of your model.

        """
        correct = Counter()
        total = Counter()
        main_acc = 0
        main_cnt = 0
        
        for i in range(len(self.test_df)):
            q = self.test_df.loc[i]["Question"].split(":")[1].split("|")[0].strip()
            gold_label = self.test_df.loc[i]['Label']
            if prediction_function_name(q) == gold_label:
                correct[gold_label] += 1
                main_acc += 1
            total[gold_label] += 1
            main_cnt += 1
        accs = {}
        for label in self.ls_labels:
            accs[label] = (correct[label]/total[label])*100
            
        return accs, 100*main_acc/main_cnt
        

    def get_sentence_representation(self, sentence):
        """
        Gives the average word2vec representation of a sentence.

        """
        words = sentence.lower().split()
        vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(300)


    def init_ml_classifier(self):
        """
        Initializes the ML classifier.

        """
        self.classifier = sklearn.linear_model.LogisticRegression(max_iter=1000)
        


    def train_label_ml_classifier(self):
        """
        Train the classifier.
        
        """
        X_train = [self.get_sentence_representation(q) for q in self.train_df['Question']]
        y_train = self.train_df['Label']
        
        self.classifier.fit(X_train, y_train)


    
    def predict_label_using_ml_classifier(self, question):
        """
        Predicts the label of the question using the classifier.

        """
        sentence_vector = self.get_sentence_representation(question).reshape(1, -1)
        return self.classifier.predict(sentence_vector)[0]


class MusicAsstSlotPredictor:
    def __init__(self):
        """
        Slot Predictor for the Music Assistant.
        """
        self.parser_files = "data/semantic-parser"
        self.train_data = []
        self.test_questions = []
        self.test_answers = []

        self.slot_names = set()

    def load_data(self):
        """
        Load the data from file.

        """
        with open(f'{self.parser_files}/music_asst_train.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.train_data.append(json.loads(line))

        with open(f'{self.parser_files}/music_asst_val_ques.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.test_questions.append(json.loads(line))

        with open(f'{self.parser_files}/music_asst_val_ans.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.test_answers.append(json.loads(line))
    
    def get_slots(self):
        """
        Get all the unique slots.

        """
        for sample in self.train_data:
            for slot_name in sample['slots']:
                self.slot_names.add(slot_name)
    
    def predict_slot_values(self, question):
        """
        Predicts the values for the slots.

        """

        words = question.split()
        slot_patterns = {
            'playlist': r'(?:my|the|a)?\s*playlist(?: named)?\s*([\w\s&\'-]+)',
            'music_item': r'(?:add|put|include)? ?(?:the |a |this )?([\w\s&]+)? ?(album|song|track|tune)?',
            'entity_name': r'add ([\w\s&]+) to',
            'playlist_owner': r'\b(my|their|his|her)\b',
            'artist': r'(?:by|from|put|add) ([\w\s&]+)'
        }
        
        slot_values = {slot: None for slot in self.slot_names}
        
        for slot_name, pattern in slot_patterns.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                slot_values[slot_name] = ' '.join(filter(None, match.groups()))
        
        return slot_values
    
    def get_confusion_matrix(self, slot_prediction_function, questions, answers):
        """
        Find the true positive, true negative, false positive, and false negative examples with respect to the prediction of a slot being active or not (irrespective of value assigned).

        """
        tp = {}
        fp = {}
        tn = {}
        fn = {}
        for slot_name in self.slot_names:
            tp[slot_name] = []
        for slot_name in self.slot_names:
            fp[slot_name] = []
        for slot_name in self.slot_names:
            tn[slot_name] = []
        for slot_name in self.slot_names:
            fn[slot_name] = []
        for i, question in enumerate(questions):
            predicted_slots = slot_prediction_function(question)
            gold_slots = answers[i]['slots']
            
            for slot in self.slot_names:
                
                predicted_active = predicted_slots[slot] is not None
                actual_active = slot in gold_slots
                
                if predicted_active and actual_active:
                    tp[slot].append(i)
                elif predicted_active and not actual_active:
                    fp[slot].append(i)
                elif not predicted_active and not actual_active:
                    tn[slot].append(i)
                elif not predicted_active and actual_active:
                    fn[slot].append(i)
                    
        return tp, fp, tn, fn
    
    def evaluate_slot_prediction_recall(self, slot_prediction_function):
        """
        Evaluates the recall for the slot predictor

        """
        correct = Counter()
        total = Counter()
        # predict slots for each question
        for i, question in enumerate(self.test_questions):
            i = self.test_questions.index(question)
            gold_slots = self.test_answers[i]['slots']
            predicted_slots = slot_prediction_function(question)
            for name in self.slot_names:
                if name in gold_slots:
                    total[name] += 1.0
                    if predicted_slots.get(name, None) != None and predicted_slots.get(name).lower() == gold_slots.get(name).lower():
                        correct[name] += 1.0
        accs = {}
        for name in self.slot_names:
            accs[name] = (correct[name] / total[name]) * 100
        return accs




class MathParser:
    def __init__(self):
        """
        Math Word Problem Solver.
        """

        self.parser_files = "data/semantic-parser"
        self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

        self.train_file = "math_train.tsv"
        self.test_file = "math_val.tsv"
        
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def load_data(self):
        """
        Load the data from file.

        """
        self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
        self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")
        
    
    def init_model(self):
        """
        Initializes the ML classifier.

        """

        self.load_data()
        questions = self.train_df["Question"].astype(str).values
        equations = self.train_df["Equation"].astype(str).values

        labels = [self.extract_operation(eq) for eq in equations]
        

        X = self.vectorizer.fit_transform(questions)
        self.model.fit(X, labels)
    

    def extract_operation(self, equation):
        """
        Extract the main operation from the equation (e.g., addition, subtraction, multiplication, division).
        """
        if '+' in equation:
            return 'addition'
        elif '-' in equation:
            return 'subtraction'
        elif '*' in equation:
            return 'multiplication'
        elif '/' in equation:
            return 'division'

    def predict_equation_from_question(self, question):
        """
        Predicts the equation for the question.

        """

        eq = ""
        
        X = self.vectorizer.transform([question])
        operation = self.model.predict(X)[0]
        numbers = re.findall(r'\d+\.?\d*', question)
        if len(numbers) < 2:
            return ""
        
        a, b = numbers[:2]
        
        if operation == 'addition':
            eq=f"{a} + {b}"
        elif operation == 'subtraction':
            eq=f"{a} - {b}"
        elif operation == 'multiplication':
            eq=f"{a} * {b}"
        elif operation == 'division':
            eq=f"{a} / {b}"
        
        return eq
    
    def ans_evaluator(self, equation):
        """
        Parses the equation to obtain the final answer.

        """
        try:
            final_ans = parse_expr(equation, evaluate = True)
        except:
            final_ans = -1000.112
        return final_ans
    
    def evaluate_accuracy(self, prediction_function_name):
        """
        Gives accuracy of model.

        """
        acc = 0
        tot = 0
        for i in range(len(self.test_df)):
            ques = self.test_df.loc[i]["Question"]
            gold_ans = self.test_df.loc[i]["Answer"]
            pred_eq = prediction_function_name(ques)
            pred_ans = self.ans_evaluator(pred_eq)

            if abs(gold_ans - pred_ans) < 0.1:
                acc += 1
            tot += 1
        return 100*acc/tot








print()
print()



print("======================================================================")
print("Checking Text2SQL Parser")
print("======================================================================")

# Define text2sql parser object
sql_parser = Text2SQLParser()

# Load the data files
sql_parser.load_data()

# Initialize the ML classifier
sql_parser.init_ml_classifier()

# Train the classifier
sql_parser.train_label_ml_classifier()

# Evaluating the keyword-based label classifier. 
print("------------- Evaluating keyword-based label classifier -------------")
accs, _ = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_keywords)
for label in accs:
	print(label + ": " + str(accs[label]))

# Evaluate the ML classifier
print("------------- Evaluating ML classifier -------------")
sql_parser.train_label_ml_classifier()
_, overall_acc = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_ml_classifier)
print("Overall accuracy: ", str(overall_acc))

print()
print()


print("======================================================================")
print("Checking Music Assistant Slot Predictor")
print("======================================================================")

# Define semantic parser object
semantic_parser = MusicAsstSlotPredictor()
# Load semantic parser data
semantic_parser.load_data()

# Look at the slots
print("------------- slots -------------")
semantic_parser.get_slots()
print(semantic_parser.slot_names)

# Evaluate slot predictor
print("------------- Evaluating slot predictor -------------")
accs = semantic_parser.evaluate_slot_prediction_recall(semantic_parser.predict_slot_values)
for slot in accs:
	print(slot + ": " + str(accs[slot]))

# Evaluate Confusion matrix examples
print("------------- Confusion matrix examples -------------")
tp, fp, tn, fn = semantic_parser.get_confusion_matrix(semantic_parser.predict_slot_values, semantic_parser.test_questions, semantic_parser.test_answers)
print(tp)
print(fp)
print(tn)
print(fn)

print()
print()


print("======================================================================")
print("Checking Math Parser")
print("======================================================================")

# Define math parser object
math_parser = MathParser()

# Load the data files
math_parser.load_data()

# Initialize and train the model
math_parser.init_model()

# Get accuracy
print("------------- Accuracy of Equation Prediction -------------")
acc = math_parser.evaluate_accuracy(math_parser.predict_equation_from_question)
print("Accuracy: ", acc)