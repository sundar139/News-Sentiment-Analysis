import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, TFBertModel
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Improved text preprocessing
lemmatizer = WordNetLemmatizer()

def improved_preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

# Load and preprocess data
filename = "News Sentiment Analysis/DATASET/News_sentiment_Jan2017_to_Apr2021.csv"
df = pd.read_csv(filename, usecols=[1, 3], names=["text", "sentiment"], encoding="utf-8", encoding_errors="replace")

# Reset the index
df = df.reset_index(drop=True)
df.dropna(inplace=True)

df['text'] = df['text'].apply(improved_preprocess_text)
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "POSITIVE" else 0)

X_text = df['text'].values
y = df['sentiment'].values
print(df.head())

# BERT Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_text = tokenizer(
    list(X_text),
    max_length=150,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

input_ids = X_text['input_ids']
attention_masks = X_text['attention_mask']

# Split data into train and test sets using TensorFlow
dataset = tf.data.Dataset.from_tensor_slices(((input_ids, attention_masks), y))
dataset = dataset.shuffle(len(y))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset = dataset.take(train_size).batch(16)
test_dataset = dataset.skip(train_size).batch(16)

# Build the BERT model
def build_model():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    input_ids = Input(shape=(150,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(150,), dtype=tf.int32, name="attention_mask")
    
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = bert_output[:, 0, :]
    
    dropout_layer = Dropout(0.4)(cls_token)
    dense_layer = Dense(64, activation='relu')(dropout_layer)
    output = Dense(1, activation='sigmoid')(dense_layer)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    optimizer = Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
model = build_model()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# No need to map again if the dataset is structured correctly
# train_dataset = train_dataset.map(lambda x, y: ((x[0], x[1]), y))
# test_dataset = test_dataset.map(lambda x, y: ((x[0], x[1]), y))

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_metrics.png')
plt.close()

# Evaluate on test set
y_pred = model.predict(test_dataset)
y_pred_labels = (y_pred > 0.5).astype(int)

# Convert y_test to numpy array
y_test = tf.concat([y for _, y in test_dataset], axis=0).numpy()

# Evaluation function
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print('\nConfusion Matrix:')
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

evaluate(y_test, y_pred_labels)
