# Topic analysis using neural networks
This repository was created for my diploma thesis. It consists of two modified repositories https://github.com/MarekUlip/TopicAnalysisDeep and https://github.com/MarekUlip/topic-analysis .

## Requirements
Scripts were written using Python 3.7  <br/>
tensorflow 2.1 gpu (view installation for details also note that scripts using generators will not run in earlier versions because some of them use fit with generators which were not supported until 2.1)

## Installation
All install lines are intended for [Anaconda](https://www.anaconda.com/distribution/) environment. So first make sure that Anaconda is up and running on your machine. <br/>
From Anaconda Prompt type following lines: <br/>
conda create --name tensorflow20 <br/>
conda activate tensorflow20
<br/>
<br/>
After activating the anaconda environment you can start installing required packages: <br/>
###Installing tensorflow
conda install -c anaconda tensorflow-gpu <br/>

### General helpful libraries
These libraries have helpful methods for analysis or showing results <br/>
conda install -c conda-forge matplotlib <br/>
conda install -c anaconda scikit-learn <br/>
<br/>
Pydot and graphviz are used to generate neural network architecture image
conda install -c anaconda pydot <br/>
conda install -c anaconda graphviz (May not be needed as pydot should be installed with it)<br/>
### Libraries for hyperparameter search
Hyperas can be installed only via pip so pip <br/>
conda install pip <br/>
pip install hyperas <br/>
### Libraries used for text analysis
conda install -c anaconda gensim <br/>
conda install -c conda-forge wordcloud <br/>
conda install -c bokeh bokeh <br/>


## Known issues
File encountered_issues.txt contains issues encountered during my work with tensorflow. 