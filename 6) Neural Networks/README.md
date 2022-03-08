Neural Network Use Cases


Neural Networks are incredibly popular and powerful machine learning models. They often perform well in cases where we have a lot of features as they automatically do feature engineering without requiring domain knowledge to restructure the features.

In this module we will be using image data. Since each pixel in the image is a feature, we can have a really large feature set. They are all commonly used in text data as it has a large feature set as well. Voice recognition is another example where neural networks often shine.
Neural networks often work well without you needing to use domain knowledge to do any feature engineering.

Biological Neural Network


A more accurate term for Neural Networks is Artificial Neural Networks (ANN). They were inspired by how biological neural networks work in human brains.

A brain’s neural network is made up of around 86 billion neurons. The neurons are connected by what are called synapses. There are about 100 trillion synapses in the human brain. The neurons send signals to each other through the synapses.
We will see in the next lessons how an artificial neural network is defined.

What's a Neuron?


An artificial neuron (often called a node) is modeled after a biological neuron. It is a simple object that can take input, do some calculations with the input, and produce an output.

We visually represent neurons as follows. x1 and x2 are the inputs. Inside the neuron, some computation is done based on x1 and x2 to produce the output y1.contentImageNeurons can take any number of inputs and can also produce any number of outputs.
Each neuron is only capable of a small computation, but when working together they become capable of solving large and complicated problems.

Neuron Computations


Inside the neuron, to do the computation to produce the output, we first put the inputs into the following equation (just like in logistic regression).contentImage
Recall that x1 and x2 are the inputs. In logistic regression, we referred to the values w1, w2, and b as the coefficients. In neural networks, we refer to w1 and w2 as the weights, and b as the bias.

We plug this value into what is called an activation function. The above equation can have a result of any real number. The activation function condenses it into a fixed range (often between 0 and 1).

A commonly used activation function is the sigmoid function, the same function we used in logistic regression. Recall that this function produces a value between 0 and 1. It is defined as follows.contentImageThe sigmoid has the following shape.contentImageTo get the output from the inputs we do the following computation. The weights, w1 and w2, and the bias, b, control what the neuron does. We call these values (w1, w2, b) the parameters. The function f is the activation function (in this case the sigmoid function). The value y is the neuron’s output.contentImage
This function can be generalized to have any number of inputs (xi) and thus the corresponding number of weights (wi).

Activation Functions


There are three commonly used activation functions: sigmoid (from the previous part), tanh, and ReLU.

Tanh has a similar form to sigmoid, though ranges from -1 to 1 instead of 0 to 1. Tanh is the hyperbolic tan function and is defined as follows:contentImage
The graph looks like this:contentImageReLU stands for Rectified Linear Unit. It is the identity function for positive numbers and sends negative numbers to 0.

Here is the equation and graph.contentImagecontentImage
Any of these activation functions will work well. Which one to use will depend on specifics of our data. In practice, we figure out which one to use by comparing the performance of different neural networks.


An Example


Assume we have a neuron that takes 2 inputs and produces 1 output and whose activation function is the sigmoid. The parameters are:

Weights (w1, w2) = [0, 1]
Bias (b) = 2

If we give the neuron input (1, 2) we get the following calculation.contentImageThe neuron yields an output of 0.9820.

Alternatively, if we give the neuron input (2, -2) we get the following calculation.contentImage
The neuron with this input yields an output of 0.5.
A neuron by itself does not have much power, but when we build a network of neurons, we can see how powerful they are together.

Multi-Layer Perceptron


To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons. We will be working with feed forward neural networks which means that the neurons only send signals in one direction. In particular, we will be working with what is called a Multi-Layer Perceptron (MLP). The neural network has multiple layers which we see depicted below.contentImageA multi-layer perceptron will always have one input layer, with a neuron (or node) for each input. In the neural network above, there are two inputs and thus two input nodes. It will have one output layer, with a node for each output. Above there is 1 output node for a single output value. It can have any number of hidden layers and each hidden layer can have any number of nodes. Above there is one hidden layer with 5 nodes.

The nodes in the input layer take a single input value and pass it forward. The nodes in the hidden layers as well as the output layer can take multiple inputs but they always produce a single output. Sometimes the nodes need to pass their output to multiple nodes. In the example above, the nodes in the input layer pass their output to each of the five nodes in the hidden layer.
A single-layer perceptron is a neural network without any hidden layers. These are rarely used. Most neural networks are multi-layer perceptrons, generally with one or two hidden layers.

Example Neural Network


Let’s dive deeper into how this works with an example. A neural network that solves any real problem will be too large to interpret, so we will walk through a simple example.

We have a neural network with two inputs, a single hidden layer with two nodes and one output. The weights and bias are given in the nodes below. All the nodes use the sigmoid activation function.contentImageLet’s see what output we get for the input (3,2).

Here is the output for the first node in the hidden layer.contentImageHere is the output for the second node in the hidden layer.contentImageHere is the output from the node in the output layer. Note that this node takes the outputs from the hidden layer as input.contentImageThus for the input (3, 2), the output of the neural network is 0.8680.
To change how the neural network performs, we can change the weights and bias values.


More Than 2 Target Values


A nice benefit of an MLP classifier is that it easily extends to problems that have more than 2 target values. In the previous modules, we have dealt with predicting 0 or 1 (true or false, survived or not, cancerous or not, ). In some cases, we will be choosing among 3 or more possible outputs. A neural network does this naturally. We just need to add more nodes to the output layer. For example, if we are trying to predict if an image is a bird, cat or dog, we will have three output nodes. The first (y1) measures if the image is a bird, the second (y2) measures if the image is a cat, and the third (y3) measures if the image is a dog. The model chooses the output with the highest value.contentImage
For example, for a given image input, the neural net outputs y1=0.3, y2=0.2 and y3=0.5, the model would then determine the image has a dog (y3) in it.
We can use any classifier for a multi-class problem, but neural networks generalize naturally.


28 Comments
Loss


In order to train a neural network, we need to define a loss function. This is a measure of how far off our neural network is from being perfect. When we train the neural network, we are optimizing a loss function.

We will use cross entropy as our loss function. This is the same as the likelihood we used in logistic regression but is called by a different name in this context. We calculate the cross entropy as follows.contentImageWe multiply together the cross entropy values for all the datapoints.

Let’s say we have two models to compare on a tiny dataset with 4 datapoints. Here is a table of the true values, the predicted probabilities for model 1 and the predicted probabilities for model 2.contentImageThe cross entropy for model 1 is as follows.contentImageThe cross entropy for model 2 is as follows.contentImageCross entropy will be higher the better the model is, thus since model 2 has higher cross entropy than model 1, it is the better model.
Just like we did with the likelihood function in logistic regression, we use the loss function to find the best possible model.


16 Comments
Backpropagation


A neural network has a lot of parameters that we can control. There are several coefficients for each node and there can be a lot of nodes! The process for updating these values to converge on the best possible model is quite complicated. The neural network works backwards from the output node iteratively updating the coefficients of the nodes. This process of moving backwards through the neural network is called backpropagation or backprop.

We won't go through all the details here as it involves calculating partial derivatives, but the idea is that we initialize all the coefficient values and iteratively change the values so that at every iteration we see improvement in the loss function. Eventually we cannot improve the loss function anymore and then we have found our optimal model.
Before we create a neural network we fix the number of nodes and number of layers. Then we use backprop to iteratively update all the coefficient values until we converge on an optimal neural network.

Creating Artificial Dataset


Sometimes in order to test models, it is helpful to create an artificial dataset. We can create a dataset of the size and complexity needed. Thus we can make a dataset that is easier to work with than a real life dataset. This can help us understand how models work before we apply them to messy real world data.

We will use the make_classification function in scikit-learn. It generates a feature matrix X and target array y. We will give it these parameters:

• n_samples: number of datapoints
• n_features: number of features
• n_informative: number of informative features
• n_redundant: number of redundant features
• random_state: random state to guarantee same result every time

You can look at the full documentation to see other parameters that you can tweak to change the result.

Here is the code to generate a dataset.
Here’s the code to plot the data so we can look at it visually.
from matplotlib import pyplot as plt
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()
PY
contentImage
Scikit-learn has a couple other functions besides make_classification for making classification datasets with different properties. Look at make_circles and make_moons if you want to play around with more artificial datasets.

MLPClassifier


Scikit-learn has an MLPClassifier class which is a multi-layer perceptron for classification. We can import the class from scikit-learn, create an MLPClassifier object and use the fit method to train.

Try it now:
Output:
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
You will notice that we get a ConvergenceWarning. This means that the neural network needs more iterations to converge on the optimal coefficients. The default number of iterations is 200. Let’s up this value to 1000.
mlp = MLPClassifier(max_iter=1000)
PY
Now when we run this code, the neural network will converge. We can now use the score method to calculate the accuracy on the test set.

Try it now:
Neural networks are incredibly complicated, but scikit-learn makes them very approachable to use!


24 Comments
Parameters for MLPClassifier


There are a couple of parameters that you may find yourself needing to change in the MLPClassifier.

You can configure the number of hidden layers and how many nodes in each layer. The default MLPClassifier will have a single hidden layer of 100 nodes. This often works really well, but we can experiment with different values. This will create an MLPCLassifier with two hidden layers, one of 100 nodes and one of 50 nodes.
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))
PY
We saw max_iter in the previous part. This is the number of iterations. In general, the more data you have, the fewer iterations you need to converge. If the value is too large, it will take too long to run the code. If the value is too small, the neural network will not converge on the optimal solution.

We also sometimes need to change alpha, which is the step size. This is how much the neural network changes the coefficients at each iteration. If the value is too small, you may never converge on the optimal solution. If the value is too large, you may miss the optimal solution. Initially you can leave this at the default. The default value of alpha is 0.0001. Note that decreasing alpha often requires an increase in max_iter.

Sometimes you will want to change the solver. This is what algorithm is used to find the optimal solution. All the solvers will work, but you may find for your dataset that a different solver finds the optimal solution faster. The options for solver are 'lbfgs', 'sgd' and 'adam'.

Run this code in the playground and try changing the parameters for the MLPClassifier. The code uses a random_state to ensure that every time you run the code with the same parameters you will get the same output.
If you look at the docs, you can read about several more parameters that you can tune in the neural network.

The MNIST Dataset


In this lesson we will be working with a new dataset, the MNIST database of handwritten digits. NIST is the National Institute of Standards and Technology and the M stands for Modified.

This is a database of images of handwritten digits. We will build a classifier to determine which digit is in the image.

We will start with the version of the MNIST dataset that is built into scikit-learn. This has the images with only 8 by 8 pixels, so they are blurry.

Here are a couple example images:contentImageIn scikit-learn we can load the dataset using the load_digits function. To simplify the problem, we will initially only be working with two digits (0 and 1), so we use the n_class parameter to limit the number of target values to 2.
from sklearn.datasets import load_digits
X, y = load_digits(n_class=2, return_X_y=True)
PY
We can see the dimensions of X and y and what the values look like as follows.
print(X.shape, y.shape)
print(X[0])
print(y[0])
PY
Output:
(360, 64) (360,)
[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
0
We see that we have 300 datapoints and each datapoint has 64 features. We have 64 features because the image is 8 x 8 pixels and we have 1 feature per pixel. The value is on a grayscale where 0 is black and 16 is white.

To get a more intuitive view of the datapoint, reshape the array to be 8x8.
print(X[0].reshape(8, 8))
PY
Output:
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
We can see that this is a 0, though we will see in the next part that we can draw the image more clearly.

Try it now:
There are different versions of this dataset with more pixels and with colors (not grayscale). We will see that even with these simplified images, we can build a good classifier.

Drawing the Digits


You can build a model without ever looking at a visual representation of the images, but it can sometimes be helpful to draw the image.

We use the matplotlib function matshow to draw the image. The cmap parameter is used to indicate that the image should be in a grayscale rather than colored.
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.show()
PY
Output:contentImage
You can see that with only 64 pixels the image is very pixelated. Even with these blurry images we can build an excellent model.

MLP for MNIST Dataset


Now let’s use the MLPClassifier to build a model for the MNIST dataset.

We will do a train/test split and train an MLPClassifier on the training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
PY
We do not get a warning, so the default number of iterations is adequate in this case.

Let’s look at how the model predicts the first datapoint in the test set. We use matplotlib to draw the images and then show the model’s prediction.
x = X_test[0]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 0
PY
Output:contentImageWe can see that this is a 0 and that our model correctly predicts 0.

Similarly, let’s look at the second datapoint.
x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 1
PY
Output:contentImageThis is clearly a 1 and our model again gets the correct prediction.

We can also see the model’s accuracy on the entire test set.
We can also see the model's accuracy on the entire test set. Thus our model gets 100% accuracy.
0 and 1 are two of the easier digits to distinguish, but we will see that the model can also perform well with distinguishing harder examples.

Classifying all 10 Digits


Since neural networks easily generalize to handle multiple outputs, we can just use the same code to build a classifier to distinguish between all ten digits.

This time when we load the digits, we do not limit the number of classes.
So we got 96% of the datapoints in the test set correct. Let’s look at the ones we got incorrect. We use a numpy mask to pull out just the datapoints we got incorrect. We pull the x values, the true y value as well as the predicted value.
y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test] 
PY
Let’s look at the first image that we got wrong and what our prediction was.
j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])
PY
Output:contentImagetrue value: 4
predicted value: 9

You can see from looking at the image that a human might also be confused. It is not obvious whether it is a 4 or a 9.

Try it now:
You can modify the code to see all of the datapoints the model predicted incorrectly.

Open ML


For this lesson, we will use a more granular version of the MNIST dataset. Instead of using the version in scikit-learn which has 64 pixel images, we will use a version from Open ML that has 784 pixels (28 x 28).

Open ML (www.openml.org) has a database of large datasets that can be used for a variety of machine learning problems. Scikit-learn has a function fetch_openml for directly downloading datasets from the Open ML database.

Use the following code to get our dataset.
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
PY
We can briefly look at the shape of the arrays, the range of the features values, and the first few values of the target array to better understand the dataset.
print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])
PY
Output:
(70000, 784) (70000,)
0.0 255.0
['5' '0' '4' '1' '9']
We can see that we have 70,000 datapoints with 784 features. The feature values range from 0 to 255 (which we interpret on a gray scale with 0 being white and 255 being black). The target values are the numbers 0-9. Note that the target values are stored as strings and not integers.

For our example, we will be using only the digits 0-3, so we can use the following code to segment out that portion of the dataset.
X5 = X[y <= '3']
y5 = y[y <= '3']
PY
We will be modifying some of the default parameters in the MLPClassifier to build the model. Since our goal will be to visualize the weights of the hidden layer, we will use only 6 nodes in the hidden layer so that we can look at all of them. We will use 'sgd' (stochastic gradient descent) as our solver which requires us to decrease alpha (the learning rate).
mlp=MLPClassifier(
  hidden_layer_sizes=(6,), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X5, y5)
PY
If we run this code we will see that it converges.
Since this dataset is quite large, you will want to work with it on your computer rather than the code playground.

MLPClassifier Coefficients


The MLPClassifier stores the coefficients in the coefs_ attribute. Let’s see what it looks like.
print(mlp.coefs_)
PY
Output:
[array([[-0.01115571, -0.08262824, 0.00865588, -0.01127292, -0.01387942,
        -0.02957163],
...
First we see that it is a list with two elements.
print(len(mlp.coefs_))
PY
Output:
2
The two elements in the list correspond to the two layers: the hidden layer and the output layer. We have an array of coefficients for each of these layers. Let’s look at the shape of the coefficients for the hidden layer.
print(mlp.coefs_[0].shape)
Output:
(784, 6)
PY
We see that we have a 2-dimensional array of size 784 x 6. There are 6 nodes and 784 input values feeding into each node, and we have a weight for each of these connections.
In order to interpret the values, we will need to use a visual representation.

Visualizing the Hidden Layer


To get a better understanding of what the neural network is doing, we can visualize the weights of the hidden layer to get some insight into what each node is doing.

We will use the matshow function from matplotlib again to draw the images. In matplotlib we can use the subplots function to create multiple plots within a single plot.
fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()
PY
Output:contentImageYou can see that nodes 4 and 6 are determining if the digit is a 3. Node 1 is determining if the digit is a 0 or a 2 since you can see both of those values in the image. Not every hidden node will have an obvious use.
If you change the random state in the MLPClassifier, you will likely get different results. There are many equivalently optimal neural networks that work differently.



Interpretability


While we can visualize the nodes in the hidden layer to understand on a high level what the neural network is doing, it is impossible to answer the question "Why did datapoint x get prediction y?" Since there are so many nodes, each with their own coefficients, it is not feasible to get a simple explanation of what the neural network is doing. This makes it a difficult model to interpret and use in certain business use cases.
Neural Networks are not a good option for interpretability.

Computation


Neural networks can take a decent amount of time to train. Each node has its own coefficients and to train they are iteratively updated, so this can be time consuming. However, they are parallelizable, so it is possible to throw computer power at them to make them train faster.
Once they are built, neural networks are not slow to make predictions, however, they are not as fast as some of the other models.

Performance


The main draw to neural networks is their performance. On many problems, their performance simply cannot be beat by other models. They can take some tuning of parameters to find the optimal performance, but they benefit from needing minimal feature engineering prior to building the model.

A lot of simpler problems, you can achieve equivalent performance with a simpler model like logistic regression, but with large unstructured datasets, neural networks outperform other models.
The key advantage of neural networks is their performance capabilities.

