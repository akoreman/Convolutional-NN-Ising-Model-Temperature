# -*- coding: utf-8 -*-
import csv as csv
import numpy as np
import matplotlib.pyplot as plt

def ReadCSV(fileLocation):
    """
    Function to read the CSV data files, input is the name of the CSV file, output nparray of vectors.
    
    (string) -> nparray of dimension # objects in dataset * length of each vector in dataset.
    """
    with open(fileLocation,'r') as csvFile:
        output = []
        csvReader = csv.reader(csvFile)

        for csvRow in csvReader:
            output.append(csvRow)
            
        return np.array(output).astype(np.double)
    
def PlotLattice(lattice, latticeSize, save = False, fileName = ""):
    """
    Function to plot the size x size vector of a dataset as a digit in a matplotlib heatplot.
    
    (list of length size x size, size) -> void.
    """
    plt.imshow(lattice.reshape((latticeSize,latticeSize)), cmap='Greys', interpolation='nearest')
    
    if save:
        plt.savefig(fileName + ".pdf")
    plt.show()