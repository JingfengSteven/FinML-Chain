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
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    

    

   
</head>
<body>

<table>
    <caption>Variable Description</caption>
    <tr>
        <th>Variable Name</th>
        <th>Description</th>
        <th>Unit</th>
        <th>Type</th>
    </tr>
    <tr>
        <td>timestamp</td>
        <td>Recording of the time of each block</td>
        <td></td>
        <td>String</td>
    </tr>
    <tr>
        <td>number</td>
        <td>The number of blocks on the chain</td>
        <td></td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>gas_used</td>
        <td>Actual Gas Used</td>
        <td>Gwei</td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>gas_limit</td>
        <td>The maximum allowed gas per block</td>
        <td>Gwei</td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>base_fee_per_gas</td>
        <td>The base fee set for each block</td>
        <td>Ether</td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>gas_fraction</td>
        <td>Fraction between Gas Used and Gas Limit</td>
        <td></td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>gas_target</td>
        <td>The optimal gas used for each block</td>
        <td></td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>Y</td>
        <td>Normalized Gas Used</td>
        <td></td>
        <td>Numeric</td>
    </tr>
    <tr>
        <td>$Y_t$</td>
        <td>Response variable equals to the gas_fraction</td>
        <td></td>
        <td>Numeric</td>
    </tr>
</table>

</body>
</html>


# Hypothesis Development
## Machine Learning Algorithm Selection
The machine learning we selected are Linear Regression, Deep Neural Network, XGBoost, and long-term Memory.
# The Machine Learning Workflow
## Model Development
### Data Selection
<p>We query Ethereum’s data using Google BigQuery. The raw data contains information on timestamps, block numbers, hash, parent hash, transaction, etc. Since our research aims to predict gas used in the next block, we only keep the relevant features, including time stamp, block number, gas limit, gas used, and base fee. Notably, the NFT-airdrop can substantially boost recipients' and non-recipients' engagement levels in the transactions. As a result, high volatility in gas used will occur and lead to subsequent base fee alternation. Hence, our research is structured around two distinct periods. The first period spans the apex of the ARB airdrop, recognized as the most substantial in 2023, from March 21 to April 1, encompassing 78290 blocks. The second period pertains to the month devoid of significant NFT-airdrop activities, spanning from June 1st, 2023, to July 1st, 2023, and containing 213244 blocks.</p>

### Data Processing

<p>In the original dataset, the base fee is denominated in units of Gwei, where each Gwei is equivalent to <code>$10^{-9}$</code> Ether. Consequently, for enhanced interpretability of the dataset, we scale the base fee by <code>$10^{-9}$</code>, expressing it in terms of Ether.</p>

<p>We create a regressor, denoted as <code>$\alpha$</code>, by computing the ratio of gas used to the gas limit. The predicted variable <code>$Y$</code> represents the normalized gas used, determined by the formula:</p>

<blockquote>
    <p>
        \[ Y = \frac{{\text{{gasUsed}} - \text{{gasTarget}}}}{{\text{{gasTarget}}}} \]
    </p>
</blockquote>

<p>For varying periods <code>$k$</code>, the regressor variable for the preceding <code>$k$</code> data points is collected into a list, forming the feature set <code>$X$</code>. The variable <code>$Y$</code> corresponds precisely to the prediction variable for the data point at time <code>$t$</code>.</p>

## Results Presentation

### Training and Testing
We conduct the train-test split through k-fold cross-validation with k=5, where the whole data set is split into 5 subsets for training, validating, and testing the model. Where the fraction of training data is 0.44, the fraction of validation data is 0.22, and the testing data is 0.34

## Model Evaluation
### Evaluation Criteria
The model is evaluated by Mean Squared Error.
### Iterative Improvement
The hyperparameter setting is the initial setting learning rate=0.001, while patience is 10, and early stopping is 12 epochs without a decrease of loss. (For both DNN and LSTM)
