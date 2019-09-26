## shapSD

**shapSD** is an interpretable framework that enables to inspect variable influence in black box models through pattern mining. Despite the global interpretation and local interpretation, this framework provides a "pattern level" interpretation, which combines the local interpretation methods and the subgroup discovery technique. 

### Global Interpretation methods

- **Partial Dependence Plot (PDP)**
    - show the marginal effect of a feature on the model predictions
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

### Subgroup Discovery technique

Subgroup Discovery is a well established data mining technique that allows you to identify patterns in your data. The goal of subgroup discovery is to identify descriptions of data subsets that show an interesting distribution with respect to a pre-specified target concept.



### ShapSD Usage

- **Tabular Data**:

[Case study: Breast Cancer](https://github.com/XiaoqiMa/shapSD/doc/03-Case Study Breast Cancer dataset.ipynb)

[Case study: Adult Income](https://github.com/XiaoqiMa/shapSD/doc/04-Case Study Adult dataset.ipynb)

[Comparison of decision tree and subgroup discovery](https://github.com/XiaoqiMa/shapSD/doc/02-Comparison between decision tree and subgroup discovery.ipynb)

- **Text Data**:

[Case study: Amazon Review](https://github.com/XiaoqiMa/shapSD/doc/05-Amazon review explanation.ipynb)

[Case study: Review sentiment](https://github.com/XiaoqiMa/shapSD/doc/06-Review sentiment explanation.ipynb)

- **Neural Networks**:

[Case study: Review sentiment](https://github.com/XiaoqiMa/shapSD/doc/07-Neural Networks.ipynb)

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
