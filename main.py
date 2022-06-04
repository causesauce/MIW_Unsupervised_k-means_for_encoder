from keras.datasets import mnist
from keras import layers
from keras import models
from keras.models import Sequential


# %% create model function
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(16, (3, 3), padding='same', activation='relu'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    model.summary()
    return model


# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# %%
model = create_model(input_shape=(28, 28, 1))
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit(train_images, train_images, epochs=2, batch_size=64)
model.save_weights('weights')

# %%
model2 = create_model((28, 28, 1))
model2.load_weights('weights')
model2.pop()
model2.pop()
model2.pop()
model2.pop()
model2.summary()

# %%
result2 = model2.predict(train_images[:5000])
print(f'result = {result2.shape}')
a, b, c, d = result2.shape
code = result2.reshape(a, b * c * d)
print(f'code = {code.shape}')
print(code)

X = code
y = train_labels[:5000]

# %% knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print(knn.score(X, y))

# %% k-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=5)
kmeans.fit(X)

# %%
distinct_labels = {i: [] for i in range(10)}
for i in range(5000):
    distinct_labels[y[i]].append(i)

predicted_labels = kmeans.labels_

#%%

predictions = kmeans.predict(X)
confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]

for label in distinct_labels.keys():
    for index in distinct_labels[label]:
        cluster = predictions[index]
        confusion_matrix[cluster][label] += 1

purity = 0
for i in range(10):
    purity += max(confusion_matrix[i])
purity /= 5000

print(f"purity of 10-clustered k-means algorithm for MNIST dataset: {purity}")

