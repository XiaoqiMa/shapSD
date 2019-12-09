## shapSD

**shapSD** is an interpretable framework that enables to inspect variable influence in black box models through pattern mining. Despite the global interpretation and local interpretation, this framework provides a meso-level interpretation, which combines the local interpretation methods and the subgroup discovery technique. 

### Global Interpretation methods

- **Permutation feature importance**
    - measured by by the drop of prediction accuracy of the model after permuting the selected feature
- **SHAP feature importance**
    - based on the magnitude of feature contribution using shapley values (estimated by the mean absolute shapley values)

### Local Interpretation methods

- **Binary feature value flip**
    - to see the effect of a binary feature after we flip the feature value
- **Numeric feature value perturb**
    - to see the effect of a numeric feature after we perturb the feature value
- **LIME** 
    - to train an interpretable model to approximate the predictions of the underlying black box model
- **shapley values**
    - to calculate the individual contribution of each feature in an instance to compose the final prediction (from coalition game theory)
- **Kernel SHAP**
    - combination of linear LIME and shapley values to get an explanation model

### Meso-level Interpretatoin methods

- **Local methods** + **Pattern Mining**


### ShapSD Usage

- **Tabular Data**:

[Case study: Adult Income](https://github.com/XiaoqiMa/shapSD/blob/master/doc/03-Case%20Study%20Adult%20dataset-boosting_tree_model.ipynb)

[Comparison of decision tree and subgroup discovery](https://github.com/XiaoqiMa/shapSD/blob/master/doc/02-Comparison%20between%20decision%20tree%20and%20subgroup%20discovery.ipynb)

- **Text Data**:

[Case study: Amazon Review](https://github.com/XiaoqiMa/shapSD/blob/master/doc/05-Amazon%20review%20explanation.ipynb)

[Case study: Review sentiment](https://github.com/XiaoqiMa/shapSD/blob/master/doc/06-Review%20sentiment%20explanation.ipynb)

- **Neural Networks**:

[Multilayer Perceptron](https://github.com/XiaoqiMa/shapSD/blob/master/doc/04-Case%20Study%20Adult%20dataset-keras_neural_network.ipynb)
[LSTM](https://github.com/XiaoqiMa/shapSD/blob/master/doc/07-Neural%20Networks.ipynb)


**The MIT License (MIT)**

    Copyright (c) 2019 Xiaoqi Ma
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
