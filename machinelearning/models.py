import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    flag = False
            if flag:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # only have 2 layer for now
        self.layer1_dimension = 20
        self.w_1 = nn.Parameter(1, self.layer1_dimension)  # input features, output features
        self.b_1 = nn.Parameter(1, self.layer1_dimension)   # 1 * input features

        self.layer2_dimension = 20
        self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
        self.b_2 = nn.Parameter(1, self.layer2_dimension)

        self.layer3_dimension = 1
        self.w_3 = nn.Parameter(self.layer2_dimension, self.layer3_dimension)  # input features, output features
        self.b_3 = nn.Parameter(1, self.layer3_dimension)

        self.batch_size = 0
        self.out_dimension = 1
        self.alpha = -0.01


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        def linear_t(x, w, b):
            return nn.AddBias(nn.Linear(x, w), b)

        first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
        second_layer = nn.ReLU(linear_t(first_layer, self.w_2, self.b_2))
        return linear_t(second_layer, self.w_3, self.b_3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = 0
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                flag += nn.as_scalar(loss)
                origin = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    # print("-------------------")
                    # print(n)
                    # print(i)
                    origin[i].update(grad[i], self.alpha)
                # if loss.data < 0.00001:
                #     flag = True
                #     break
            if flag / dataset.x.shape[0] < 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 0
        self.in_dimension = 784
        self.out_dimension = 10
        self.alpha = -0.01
        self.setting = 1

        if self.setting == 1:
            self.layer1_dimension = 100
            self.w_1 = nn.Parameter(self.in_dimension, self.layer1_dimension)  # input features, output features
            self.b_1 = nn.Parameter(1, self.layer1_dimension)  # 1 * output features

            self.layer2_dimension = 70
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

            self.layer3_dimension = self.out_dimension
            self.w_3 = nn.Parameter(self.layer2_dimension, self.layer3_dimension)  # input features, output features
            self.b_3 = nn.Parameter(1, self.layer3_dimension)

        else:
            self.layer1_dimension = 100
            self.w_1 = nn.Parameter(self.in_dimension, self.layer1_dimension)  # input features, output features
            self.b_1 = nn.Parameter(1, self.layer1_dimension)  # 1 * output features

            self.layer2_dimension = self.out_dimension
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        def linear_t(x, w, b):
            return nn.AddBias(nn.Linear(x, w), b)

        if self.setting == 1:
            first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
            second_layer = nn.ReLU(linear_t(first_layer, self.w_2, self.b_2))
            return linear_t(second_layer, self.w_3, self.b_3)
        else:
            first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
            return linear_t(first_layer, self.w_2, self.b_2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                if self.setting == 1:
                    origin = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                else:
                    origin = [self.w_1, self.b_1, self.w_2, self.b_2]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    origin[i].update(grad[i], self.alpha)
            if dataset.get_validation_accuracy() > 0.96:
                self.alpha = -0.003
            if dataset.get_validation_accuracy() > 0.972:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 0
        self.in_dimension = self.num_chars
        self.out_dimension = len(self.languages)
        self.alpha = -0.01
        self.setting = 1

        if self.setting == 1:
            self.h_layer1_dimension = 100
            self.w = nn.Parameter(self.in_dimension, self.h_layer1_dimension)  # input features, output features
            self.w_hidden = nn.Parameter(self.h_layer1_dimension, self.h_layer1_dimension)
            self.w_output = nn.Parameter(self.h_layer1_dimension, self.out_dimension)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = xs[0].data.shape[0]
        f = nn.ReLU(nn.Linear(xs[0], self.w))
        for i in range(len(xs)):
            if i == 0:
                f = nn.ReLU(nn.Linear(xs[0], self.w))
            else:
                f = nn.ReLU(nn.Add(nn.Linear(xs[i], self.w), nn.Linear(f, self.w_hidden)))
        return nn.Linear(f, self.w_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                origin = [self.w, self.w_hidden, self.w_output]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    origin[i].update(grad[i], self.alpha)
            if dataset.get_validation_accuracy() > 0.8:
                self.alpha = -0.003
            if dataset.get_validation_accuracy() > 0.87:
                break
