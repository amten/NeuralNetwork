## What's this?

Java (convolutional or fully-connected) neural network implementation with plugin for [Weka] (http://www.cs.waikato.ac.nz/ml/weka/). Uses dropout and rectified linear units. Implementation is multithreaded and uses [MTJ] (https://github.com/fommil/matrix-toolkits-java) matrix library with native libs for performance.

## Installation

### Weka

Go to https://github.com/amten/NeuralNetwork/releases/latest to find the latest release. Download the files NeuralNetwork.zip and BLAS-dlls.zip. 
In Weka, go to Tools/Package Manager and press the "File/URL" button. Browse to the NeuralNetwork.zip file and press "ok".

**Important!** For optimal performance, you need to install native matrix library files.  
Windows: Unzip the BLAS-dlls.zip file to Wekas install dir (".../Program Files/Weka-3-7").  
Ubuntu: Run "sudo apt-get install libatlas3-base libopenblas-base" in a terminal window.

### Standalone

This package was made mainly to be used from the Weka UI, but it can be used in your own java code as well.

Go to https://github.com/amten/NeuralNetwork/releases/latest to find the latest release. Download the file NeuralNetwork.zip and unzip. 

Include the files NeuralNetwork.jar, lib/mtj-1.0-snapshot.jar, lib/opencsv-2.3.jar in your classpath.

**Important!** For optimal performance, you need to install native matrix library files.  
Windows: Unzip the BLAS-dlls.zip file to the directory where you execute your application, or any other directory in the PATH.  
Ubuntu: Run "sudo apt-get install libatlas3-base libopenblas-base" in a terminal window.

## Usage

### Weka

In Weka, you will find the classifier under classifiers/functions/NeuralNetwork. For explanations of the settings, click the "more" button.

**Note 1**: If you start Weka with console (alternative available in the windows start menu), you will get printouts of cost during each iteration of training and you can press enter in the console window to halt the training.

**Note 2**: When using dropout as regularization, it might still be a good idea to keep a small weight penalty. This keeps weights from exploding and causing overflows.

**Note 3**: When using convolutional layers, it seems to be most efficient to use batch-size=1 (i.e. Stochastic Gradient Descent)

### Standalone

Example code showing classification and regression can be found here:
https://github.com/amten/NeuralNetwork/tree/master/src/amten/ml/examples


## License

Free to copy and modify. Please include author name if you copy code.
