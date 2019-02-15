from math import exp
import random
import numpy as np

# TODO: Calculate logistic
def logistic(x):
    log = None
    try:
        log = (exp(x)/(1+exp(x)))
    except OverflowError:
        # ans = float('inf')
        pass
    return log

# TODO: Calculate dot product of two lists
def dot(x, y):
    s = np.dot( np.array(x), np.array(y) )
    return s

# TODO: Calculate prediction based on model
def predict(model, point):
    # print(model)
    XdW = dot(model, point["features"])
    P = logistic(XdW)
    return P

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for i in range(len(predictions)):
        if (predictions[i] > 0.5) == data[i]['label']:
            correct += 1
    return float(correct)/len(data)

# TODO: Update model using learning rate and L2 regularization
def update(model, point, rate, lam):

    
    weights = np.array(model)
    try:
        P = predict(model, point)
        loss = P - point["label"]
        lossSquared = loss**2
        weights = weights + rate * (lossSquared + weights*lam)
    except TypeError:
        # print("exponent was too big")
        pass
    return weights

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    
    model = initialize_model(len(data[0]['features']))
    for i in range(epochs):
        # slowly lowers learning rate as model converges
        # print("divL ", float(i)/float(epochs))
        print(i)
        # print(epochs)
        if (float(i)/float(epochs) >= 0.7):
            # print('yunk')
            rate = 0.05
        if (float(i)/float(epochs) >= 0.9):
            # print('yunk2')
            rate = 0.02
        if (float(i)/float(epochs) >= 0.95):
            # print('yunk3')
            rate = 0.01
        # print("Rate: ", rate)

        # this suffle portion is important as SGD is trained online and not in batches
        random.shuffle(data)
        for point in data:

            model = update(model, point, rate, lam)
    return model
        
def extract_features(raw):
    ######################################
    # at no change of features and parameters 100, 0.1, we have 0.75104 acc
    # with feature: sex==Male and parameters 20, 0.1, we have 0.75081 acc
    # Above exp 50 epochs, 0.75097 acc
    # with only feature: race==White feature added, 50, 0.1, we have 0.75097 acc

    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        
        # features.append(float(r['age'])/100)
        # features.append(float(r['age'])>=45)
        # features.append( (float(r['age'])-17.) / (90.-17.) ) 
        # features.append(float(r['education_num'])/20)
        features.append(float(r['education_num'])/16.)
        # features.append(float(r['education_num'])>=11)
        features.append(float(r['capital_gain']) / 99999 )
        features.append(float(r['capital_loss']) / 4356 )
        # features.append(r['relationship'] == 'Unmarried')



        # features.append(r['marital'] == 'Married-civ-spouse')
        features.append( (float(r['hr_per_week'])-1) / 99 )
        features.append( (float(r['age'])-17.) / (90.-17.) ) 
        # features.append(r['education'] == 'Bachelors')
        features.append(r['education'] == 'Doctorate')
        features.append(r['education'] == 'Masters')
        # features.append(r['education'] == 'Prof-school')
        # features.append(r['education'] == 'Prof-school')

        
        

        # capital_loss


        # Keep Numerical


        # Additional features
        features.append(r['relationship'] == 'Husband')
        # features.append(r['sex'] == 'Male')
        # features.append(r['race'] == 'Black')
        features.append(r['race'] == 'White')
        # features.append(r['race'] == 'Hispanic')
        features.append(r['race'] == 'Asian-Pac-Islander')
        features.append(r['education'] == 'HS-grad')
        # features.append(r['marital'] == 'Never-married')
        
        

        #TODO: Add more feature extraction rules here!
        point['features'] = features
        

        data.append(point)
    # print(features)
    # print(data[0]['features'])
    return data

# TODO: Tune your parameters for final model
def modelsgd(data):
    # Learning rate is tuned automatically in train function as more epochs pass
    return train(data, 50, 0.1, 0.5)
    
