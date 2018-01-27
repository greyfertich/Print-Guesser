# Print-Guesser

Print-Guesser is a machine learning program that can predict user-drawn, uppercase letters. Using a feedforward neural network, this program learns from around 3000 training examples and can correctly read letters around 85-90% of the time.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What you will need:
-Latest version of [Python 3](https://www.python.org/downloads/)
-[Python Imaging Library](http://www.pythonware.com/products/pil/)
-[Numpy and Scipy](https://www.scipy.org/scipylib/download.html)
-[Matplotlib](https://matplotlib.org/users/installing.html)

### Installing

Start by cloning or downloading project onto system

#### Get Python 3 and all dependencies

##### Windows

* Download the latest version of [python 3](https://www.python.org/downloads/) from python.org
* Run the installer and follow through installation instructions
* Download and install [PyCharm IDE](https://www.jetbrains.com/pycharm/)
* Open the downloaded project as the project folder
* Go to File -> Settings -> Project Interpreter
* Select the downloaded python version as the interpreter
* Click on the plus button and add the numpy, scipy, matplotlib, and pillow modules to the project

##### Mac OSX

The latest version of Mac OS X, High Sierram cones with Python 2.7 out of the box. In order to install Python 3, follow these steps
* Download [XCode](https://developer.apple.com/xcode/) from Apple
* Install the Homebrew package manager by opening the Terminal and running the following command
'''
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
'''
* Insert Homebrew directory at the top of your PATH environment variable by adding the following line at the bottom of your ~/.profile file
'''
$ export PATH=/usr/local/bin:/usr/local/sbin:$PATH
'''
* Install Python 3
'''
$ brew install python3
'''
* Run the following commands to get all the necessary packages
'''
$ easy_install pip
'''
* Numpy
'''
$ pip install numpy
'''
Scipy
'''
$ brew install gfortran
$ pip install scipy
'''
* Matplotlib
'''
$ brew install pkg-config
$ pip install matplotlib
'''
* PIL
'''
$ pip install Pillow
'''

##### Linux

Most Linux distributions already have Python 3 installed

Run the following commands to install the necessary packages
* Numpy
$ sudo apt-get install python-numpy
'''
* Scipy
'''
$ sudo apt-get install python-scipy
'''
* PIL
'''
sudo apt-get install python-pillow
'''
* Matplotlib
'''
$ sudo apt-get install python-matplotlib
'''

## Deployment

Follow these steps to run the full program

### Windows
* Navigate to the file GUI_setup.py in PyCharm and press run

### Mac OSX and Linux
* Open the terminal
* Navigate to the project directory
'''
$ cd "_Enter project path here_"
'''
* Run the project
'''
$ python3 GUI_setup.py
'''

## Built With

* [Tkinter](https://wiki.python.org/moin/TkInter) - GUI framework used

## Authors

* **Aiden Grey Fertich** - [greyfertich](https://github.com/greyfertich)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This project was inspired by Tariq Rashid's book, [Make Your Own Neural Network](https://www.barnesandnoble.com/w/make-your-own-neural-network-tariq-rashid/1123691651), where a neural network was used to decipher images of handwritten numbers.
