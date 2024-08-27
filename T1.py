import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras import layers, Model, optimizers, regularizers
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
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
filename = "DATASET/News_sentiment_Jan2017_to_Apr2021.csv"
df = pd.read_csv(filename, usecols=[1, 3], names=["text", "sentiment"], encoding="utf-8", encoding_errors="replace")

# Reset the index
df = df.reset_index(drop=True)

df.dropna(inplace=True)

df['text'] = df['text'].apply(improved_preprocess_text)
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "POSITIVE" else 0)

X_text = df['text'].values
y = df['sentiment'].values
print(df.head())

# Tokenization and padding
max_words = 250000
max_len = 150
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)
X_text = tokenizer.texts_to_sequences(X_text)
X_text = pad_sequences(X_text, maxlen=max_len)

total_tokens = sum(len(tokens) for tokens in X_text)
vocabulary_size = len(tokenizer.word_index)

print("Total number of tokens:", total_tokens)
print("Size of vocabulary:", vocabulary_size)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=32)

def build_model():
    embedding_size = 100
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = layers.Embedding(max_words, embedding_size, input_length=max_len)(inputs)
    
    conv1d_layer1 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(embedding_layer)
    conv1d_layer2 = layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(embedding_layer)
    
    concat_layer = layers.Concatenate(axis=-1)([conv1d_layer1, conv1d_layer2])
    
    attention_layer = layers.Attention()([concat_layer, concat_layer])
    
    bidirectional_lstm = Bidirectional(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(attention_layer)
    
    attention_layer2 = layers.Attention()([bidirectional_lstm, bidirectional_lstm])
    
    global_avg_pooling = layers.GlobalAveragePooling1D()(attention_layer2)
    
    dropout_layer = layers.Dropout(0.4)(global_avg_pooling)
    dense_layer1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dropout_layer)
    dense_layer2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense_layer1)
    outputs = layers.Dense(1, activation='sigmoid')(dense_layer2)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = RMSprop(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = build_model()

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=128,
    epochs=20,
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
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

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