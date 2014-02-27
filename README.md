
## What's this?

Java neural network implementation with plugin for [WEKA] (http://www.cs.waikato.ac.nz/ml/weka/). Uses dropout and rectified linear units. Implementation is multithreaded and uses [MTJ] (https://github.com/fommil/matrix-toolkits-java) matrix library with native libs for performance.

## Installation

Go to https://github.com/amten/NeuralNetwork/releases/latest to find the latest release. Download the files NeuralNetwork.zip and BLAS-dlls.zip. 
In WEKA, go to Tools/Package Manager and press the "File/URL" button. Browse to the NeuralNetwork.zip file and press "ok".

**Important!** For optimal performance on Windows, you need to copy native matrix library dll-files to Wekas install dir.
Unzip the BLAS-dlls.zip file to Wekas install dir (".../Program Files/Weka-3-7").

For Linux, native matrix library files have not been tested, though it should be possible to install using instructions given [here] (https://github.com/fommil/netlib-java#linux)

## Usage

In WEKA, you will find the classifier under classifiers/functions/NeuralNetwork.

**Note 1**: If you start Weka with console (alternative available in the windows start menu), you will get printouts of cost during each iteration of training and you can press enter in the console window to halt the training.

**Note 2**: When using dropout as regularization, it might still be a good idea to keep a small weight penalty. This keeps weights from exploding and causing overflows.


## License

Free to copy and modify. Please include author name if you copy code.
