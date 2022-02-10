import numpy as np

class LogisticRegression:
  def __init__(self):
    self._weights = []
    self._learningRate = 0.01
    self._epoch = 10
    self._x = []
    self._y = []  
  
  def linearFunction(self, data, weights):
    return np.dot(data, weights)


  def logisticFunction(self, linearResult):
    return 1 / ( 1 + np.exp(-(linearResult)))

  def updateWeights(self, weights, x, y):
    temp = np.zeros(len(x))
    for i in range(len(weights)):
      d = (self.logisticFunction(self.linearFunction(x, weights)) - y) * x[i]
      temp[i] = weights[i] - (self._learningRate * d)
    
    return temp

  def logisticRegressionLoss(self, yPredicted, yActual):
    loss = -(yActual * np.log(yPredicted) + (1-yActual) * np.log(1-yPredicted))
    meanLoss = loss.mean()
    return meanLoss


  def logisticRegressionAccuracy(self, yPredicted, yActual):
    yPredicted = yPredicted.round().astype(int)
    xor = yPredicted ^ yActual
    return np.count_nonzero(xor == 0)/len(xor)

  def fit(self, x, y, learningRate, epoch):
    self._x = x
    self._y = y
    self._learningRate = learningRate
    self._epoch = epoch

    featureCount = len(self._x[0])
    self._weights = np.random.rand(featureCount)

    for epoch in range(self._epoch):
      listResult = []

      for i in range(len(self._x)):
        listResult.append(self.logisticFunction(self.linearFunction(self._x[i], self._weights)))
        self._weights = self.updateWeights(self._weights, 0.1, self._x[i], self._y[i])

      print("epoch: ", epoch + 1, 
              "\tloss: ", self.logisticRegressionLoss(np.asarray(listResult), y), 
              "\taccuracy: ", self.logisticRegressionAccuracy(np.asarray(listResult), y)
      )

  def predict(self, x):
    x = np.array(x)
    if x.ndim == 2:
      listResult = []
      for i in range(len(x)):
        listResult.append(self.logisticFunction(self.linearFunction(x[i], self._weights)))

      return listResult
    else: 
      raise ValueError("Got array in dimension of " + str(x.ndim) + ", expecting array in a dimension of 2." )
