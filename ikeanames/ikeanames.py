import os
import sys
from bisect import bisect_left
import numpy as np
import keras.models
from keras.models import Sequential
from keras.layers import Dense, LSTM

def load_names(fn=None):
    """Load and clean the list of names.
    """
    if fn is None:
        fn = os.path.dirname(os.path.abspath(__file__)) + \
            "/../data/names.txt"
    with open(fn) as f:
        names = f.readlines()
    names = [n.strip() for n in names]
    return names
    

def get_encoding(names):
    """The encoding maps characters to numeric codes.
    The decoding is its inverse.
    We use the '.' character for padding.
    """
    max_len = max(len(n) for n in names)
    all_chars = list(set("".join(names)))
    all_chars.sort() # list of all characters in the set
    num_chars = len(all_chars)
    # do not use 0 in encoding
    encoding = {all_chars[i-1]: i for i in range(1,num_chars+1)}
    encoding["."] = 0
    decoding = {v: k for k, v in encoding.items()}
    
    return (max_len, encoding, decoding)


def make_train_set(names, encoding, decoding, max_len):
    """Generates the training set. This consists of substrings from the
    names matched with the next character in the name. The characters are
    encoded into one-hot vectors.
    """

    num_chars = len(encoding)
    names_chars = []
    # Create all substrings ns and following characters c using '.'
    # characters for padding
    for n in names:
        for i in range(len(n)+1):           
            ns = "."*(max_len-i) + n[:i]
            c = n[i] if i<len(n) else "."
            names_chars.append((ns,c))

    # Convert the substring/character set into encoded form
    X = np.zeros((len(names_chars), max_len, num_chars), dtype=np.float32)
    y = np.zeros((len(names_chars), num_chars), dtype=np.float32)
    for (i,(ns,c)) in enumerate(names_chars):
        for j in range(len(ns)):            
            X[i,j,encoding[ns[j]]] = 1
        y[i,encoding[c]] = 1

    return (X,y)


def make_model(X):
    """Create the LSTM predictor.
    """
    num_chars = X.shape[2]
    model = Sequential()
    model.add(LSTM(30, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(num_chars, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', 
        optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model,X,y,epochs=100):
    """Train the model. 100 iterations should be a good number for this
    dataset.
    """
    model.fit(X,y,epochs=epochs,verbose=2)


def load_model(fn=None):
    """Load the stored model.
    """
    if fn is None:
        fn = os.path.dirname(os.path.abspath(__file__)) + \
            "/../models/model_default.h5"
    return keras.models.load_model(fn)


def save_model(model, fn=None):
    """Save the model on the disk.
    """
    if fn is None:
        fn = os.path.dirname(os.path.abspath(__file__))+"/../models/model.h5"
    model.save(fn)


def predict_name(model, max_len, encoding, decoding, mode="gen"):
    """Generate a random name using the model.
    If mode is set to "max", this instead outputs the most likely name
    according to the model.
    """
    c = ""
    n = ""
    num_chars = len(encoding)
    while c != "." and len(n)<max_len:
        # Apply padding to current string and predict next character
        ns = "."*(max_len-len(n)) + n
        X = np.zeros((1,max_len, num_chars), dtype=np.float32)
        for j in range(len(ns)):
            X[0,j,encoding[ns[j]]] = 1
        y = model.predict(X)

        if mode == "max":
            # Get the most likely character
            c = decoding[y.argmax()]
        elif mode == "gen":
            # Select a character at random according to the probabilities
            # given by the model
            y = y[0,:]
            cy = np.cumsum(y.astype(float))
            r = np.random.rand()
            ind = bisect_left(cy,r)
            c = decoding[ind]

        if c != ".":
            n += c

    return n

if __name__ == "__main__":
    names = load_names()
    (max_len, encoding, decoding) = get_encoding(names)
    mode = sys.argv[1]

    if mode == "generate":
        model = load_model()
        try: 
            n = int(sys.argv[2])
        except IndexError:
            n = 10
        for i in range(n):
            name = predict_name(model, max_len, encoding, decoding)
            if name in names:
                name += "*"
            print(name)
    elif mode == "max":
        model = load_model()
        name = predict_name(model, max_len, encoding, decoding, mode="max")
        print(name)
    elif mode == "train":
        (X,y) = make_train_set(names, encoding, decoding, max_len)
        model = make_model(X)
        train_model(model, X, y)
        try:
            fn = sys.argv[2]
        except IndexError:
            fn = None
        save_model(model, fn)
