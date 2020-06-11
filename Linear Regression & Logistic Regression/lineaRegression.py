#Class for lineaRegression calculation
#Includes: Optimal weight vector and Average squared error Calculator
#
class LineaRegression:
    def __init__(self, x, y):
        self.data = x
        self.target = y
        self.weight = ((self.data.T * self.data).I) * (self.data.T * self.target)

    # Compute the optimal weight vector (XT X)−1 XT Y .
    def weight_vector(self):
        print(self.weight)

    #Compute average squared error
    def ase(self):
        total = 0
        predict = self.data * self.weight
        for j in range(len(self.data)):
            # sum of squared error formula
            total = total + ((self.target[j] - predict[j]) ** 2)

        '''Sum of squared error normalized by
        the total number of examples in the data '''
        total = total/len(self.data)
        return total[0,0]
