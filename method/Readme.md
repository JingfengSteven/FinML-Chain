# Research Question Formulation
## Objective
RQ1: How do different machine learning methods differ in terms of their predictive accuracy when forecasting gas used? <br>
RQ2: Which machine learning approach demonstrates the best consistent predictive performance in both market stability and periods of market volatility?
## Significance 
Adopting the perspective of a mechanism designer, our objective is to elevate the precision
of predicting future gas usage by utilizing various machine-learning approaches.
This initiative offers TFM designers a more refined reference point for contemplating future gas
predictions.
# Operational Measures
## Variables
<p>In the original dataset, the base fee is denominated in units of Gwei, where each Gwei is equivalent to $10^{-9}$ Ether. Consequently, for enhanced interpretability of the dataset, we scale the base fee by $10^{-9}$, expressing it in terms of Ether.</p>

<p>We create a regressor, denoted as $\alpha$, by computing the ratio of gas used to the gas limit. The predicted variable $Y$ represents the normalized gas used, determined by the formula:</p>

<blockquote>
    <p>
        \[ Y = \frac{{\text{{gasUsed}} - \text{{gasTarget}}}}{{\text{{gasTarget}}}} \]
    </p>
</blockquote>

<p>For varying time spans \(k\), the regressor variable for the preceding \(k\) data points is collected into a list, forming the feature set \(X\). The variable \(Y\) corresponds precisely to the prediction variable for the data point at time \(t\).</p>

# Hypothesis Development
## Data Type
## Justification
## Machine Learning Algorithm Selection

# The Machine Learning Workflow
## Model Development
### Data Processing
## Results Presentation
### Training and Testing
### Data Visualization
## Model Evaluation
### Evaluation Criteria
### Iterative Improvement
