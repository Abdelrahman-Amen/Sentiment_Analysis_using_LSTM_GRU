# Sentiment Analysis on IMDB Dataset using LSTM and GRU 📽

# Project Overview  📝

This project aims to perform sentiment analysis on the IMDB movie reviews dataset. Sentiment analysis involves classifying text into categories such as positive or negative based on the sentiment conveyed in the text. This project uses LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architectures, both of which are commonly used in Natural Language Processing (NLP) tasks for their ability to handle sequences of text and capture long-term dependencies. The goal is to compare the performance of these two architectures on a sentiment analysis task.



# Dataset   Description  🎬
The dataset used in this project is the IMDB dataset, a large collection of movie reviews that have been labeled as either positive or negative. The IMDB dataset consists of:

• Training data: 25,000 movie reviews (12,500 positive and 12,500 negative).
• Test data: 25,000 movie reviews (12,500 positive and 12,500 negative).

# Preprocessing and Tokenization 🔄


Before training the models, the text data undergoes several preprocessing steps:

• HTML Tag Removal: HTML tags are removed to focus on the pure text content.

• Text Lowercasing: All text is converted to lowercase to maintain consistency.

• Punctuation Removal: Punctuation marks are removed as they do not contribute to sentiment analysis.

• Tokenization: Words are split from the text, creating a list of words.

• Stopwords Removal: Common words (like "the", "and", etc.) that do not add meaningful information are removed.

• Lemmatization: Words are reduced to their root form (e.g., "running" becomes "run").

• Text Vectorization: The processed text is converted into numerical representations using tokenization and padding to ensure uniform input sizes.

# LSTM (Long Short-Term Memory) Architecture 🧠
![download](https://github.com/user-attachments/assets/0b3889a3-743f-451f-8414-605e84a064bd)


LSTM is a type of recurrent neural network (RNN) designed to combat the vanishing gradient problem, which occurs when training traditional RNNs on long sequences. LSTMs are particularly effective for tasks like sentiment analysis, where the order and context of words in a sentence are important.

## Components of the LSTM Model:
1. Embedding Layer: This layer converts words into dense vectors of fixed size. It learns the relationships between words during the training process.
  
2. SpatialDropout1D: A regularization technique to prevent overfitting by randomly setting entire 1D feature maps to zero during training.
 
3. LSTM Layer: The core of the model, which processes sequences of text. It learns to capture long-term dependencies in the data.
 
4. Dropout Layer: Another regularization technique that randomly sets input units to zero to prevent overfitting.
 
5. BatchNormalization: A technique to stabilize and speed up training by normalizing the output of previous layers.
 
6. Dense Layer: The final layer of the model, which outputs a single value between 0 and 1 representing the sentiment (positive or negative). A sigmoid activation function is used to output probabilities.


## Why LSTM Works Well:
• LSTM networks are capable of learning and remembering long-term dependencies in sequential data, which is crucial for understanding context in a sentence.

• In sentiment analysis, words in a sentence can affect the meaning of other words that may be far away. LSTM's ability to capture these long-term dependencies makes it well-suited for this task.



# GRU Architecture (Gated Recurrent Unit) ⚡
![gated](https://github.com/user-attachments/assets/02c2f946-5cc8-4acc-983e-d5353866d3ba)


GRU is another type of recurrent neural network that is similar to LSTM but with a simplified structure. It was designed to be computationally more efficient than LSTM while still maintaining its ability to capture long-term dependencies.

## Components of the GRU Model: 

1. Embedding Layer: Just like in the LSTM model, the embedding layer converts words into dense vectors.
 
2. SpatialDropout1D: Used for regularization to reduce overfitting.
 
3. GRU Layer: The core of the GRU model, which is similar to LSTM but with fewer parameters, making it faster and more efficient in terms of computation.
 
4. Dropout Layer: Helps prevent overfitting by randomly deactivating units.
 
5. BatchNormalization: Stabilizes the training process.
 
6. Dense Layer: Outputs the final sentiment classification (positive or negative).

## Why GRU Works Well:

• GRU performs similarly to LSTM in capturing long-term dependencies but has fewer gates, making it computationally more efficient and faster to train.

• GRU is also easier to tune and often performs well with fewer parameters, making it suitable for a wide range of tasks, including sentiment analysis.


![compare 2](https://github.com/user-attachments/assets/2e1fcd7d-4c04-420c-83ee-7e560973cd70)

# Model Training and Evaluation ⚙️

Both LSTM and GRU models were trained on the pre-processed IMDB dataset. The training process involved:

• Splitting the data into training and validation sets. 

• Using binary cross-entropy as the loss function, as the task is binary classification (positive or negative sentiment).

• Optimizing the models with the Adam optimizer, which is an adaptive learning rate optimization algorithm.

• Training for 10 epochs with a batch size of 32.

After training, the models were evaluated using metrics such as accuracy and classification report, which includes precision, recall, and F1-score. The models were also evaluated on their ability to predict sentiment on the validation dataset.


# Results 📈
• LSTM Model: The LSTM model achieved a good balance between training and validation accuracy, demonstrating its ability to capture complex sequential patterns in text.


• GRU Model: The GRU model performed similarly to the LSTM model but with faster training times and fewer parameters.





# Accuracy and Classification Report:
• Accuracy: Both models achieved high accuracy on the validation data, indicating their effectiveness in sentiment classification.

• Classification Report: The classification report showed that both models performed well in terms of precision, recall, and F1-score for both positive and negative sentiment.


# Conclusion  🏁
 

In this project, both LSTM and GRU architectures were applied to the task of sentiment analysis on the IMDB dataset. The key findings include:

• LSTM: Works well for capturing long-term dependencies and complex patterns in text but requires more computational resources due to its larger architecture.

• GRU: Provides a more efficient alternative to LSTM with comparable performance, making it suitable for tasks where computational efficiency is crucial.

## Overall, both architectures are effective for sentiment analysis tasks, with GRU being a more lightweight option. Depending on the specific requirements of a project (e.g., computational resources vs. accuracy), either LSTM or GRU could be chosen.
