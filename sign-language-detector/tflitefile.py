import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os


os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Use label encoder to convert categorical labels into numeric format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Define and train RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Evaluate RandomForestClassifier
y_predict_rf = rf_model.predict(x_test)
score_rf = accuracy_score(y_predict_rf, y_test)
print('Random Forest Classifier Accuracy:', score_rf)

# Define a simple feedforward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    tf.compat.v1.losses.sparse_softmax_cross_entropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)


# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
# Evaluate the model
test_results = model.evaluate(x_test, y_test)
print(f'Test Loss (TensorFlow model): {test_results[0]}')
print(f'Test Accuracy (TensorFlow model): {test_results[1]}')


# Save the TensorFlow model
model.save('tensorflow_model.h5')

# Load your Keras model




