# Research Question Formulation
## Objective
RQ1:  Can this dataset be applied to different machine learning models? <br>
RQ2: Can this dataset be extended to apply to other algorithms, enhancing its utility and revealing additional insights in financial market analysis?
## Significance 

Our analysis indicates that among the four machine learning models evaluated, DNN demonstrated the best prediction performance in both fungible token airdrop and normal periods. Specifically, DNN models showed superior predictive accuracy for gas usage during these periods. The results highlighted minimal disparity in prediction loss when comparing a DNN trained on data from normal periods to a DNN trained on data specifically from the token-airdrop period. This finding suggests that DNN models exhibit robust scalability on this dataset, eliminating the need to train a new neural network specifically for token-airdrop events. In addition to evaluating DNN models, we explored the extensibility of incorporating monotonicity constraints and sentiment analysis within the Neural Additive Model (NAM). Although these enhancements did not significantly improve the predictive accuracy of the NAM model on our test dataset, the intrinsic variability and complexity of blockchain data imply that different datasets from different time periods might yield different results. This opens a significant platform for other researchers to utilize and further explore the dataset, enabling comprehensive analyses and advancing financial machine-learning models. Our contributions include developing a comprehensive dataset that integrates both on-chain and off-chain data, compatible with various machine learning algorithms for financial prediction. This dataset forms the cornerstone of a novel research framework, enabling a deeper exploration of the financial market and its mechanisms. By offering a robust and versatile dataset, we facilitate advanced exploration and optimization efforts, driving innovation and enhancing the accuracy and reliability of financial machine-learning models in blockchain technology.


### Data Selection
<p>We query Ethereum’s data using Google BigQuery. The raw data contains information on timestamps, block numbers, hash, parent hash, transaction, etc. Since our research aims to predict gas used in the next block, we only keep the relevant features, including time stamp, block number, gas limit, gas used, and base fee. Notably, the token-airdrop can substantially boost recipients' and non-recipients' engagement levels in the transactions. As a result, high volatility in gas used will occur and lead to subsequent base fee alternation. Hence, our research is structured around two distinct periods. The first period spans the apex of the ARB airdrop, recognized as the most substantial in 2023, from March 21 to April 1, encompassing 78290 blocks. The second period pertains to the month devoid of significant token-airdrop activities, spanning from June 1st, 2023, to July 1st, 2023, and containing 213244 blocks.</p>

<p>We also query the discord data. 

  
### Data 
#### On-chain data
We utilize Google BigQuery to extract Ethereum's blockchain data, including timestamps, block numbers, hashes, parent hashes, transactions, etc. We retain only the pertinent features to predict gas usage in forthcoming blocks: timestamp, gas limit, gas used, and base fee. We exclude other variables, such as transaction numbers, despite their high correlation with gas usage, based on our specific research focus. Furthermore, our study acknowledges the impact of token airdrops on transaction engagement levels for recipients and non-recipients. According to Guo\cite{guo2023spillover}, token airdrops can significantly influence engagement, resulting in pronounced gas usage volatility and subsequent base fee fluctuations. Consequently, our analysis is bifurcated into two distinct periods. The first period examines the ARB token airdrop, the most substantial airdrop event in 2023, which occurred from March 21 to April 1 and comprised 78,290 blocks. The second period, devoid of significant fungible token airdrop activities, extends from June 1, 2023, to July 1, 2023, encompassing 213,244 blocks. This temporal delineation allows for a comprehensive analysis of the effects of significant airdrop events on Ethereum's gas dynamics.

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

#### Data processing
<p>In the original dataset, the base fee is denominated in units of Gwei, where each Gwei is equivalent to <code>$10^{-9}$</code> Ether. Consequently, for enhanced interpretability of the dataset, we scale the base fee by <code>$10^{-9}$</code>, expressing it in terms of Ether.</p>

<p>We create a regressor, denoted as <code>$\alpha$</code>, by computing the ratio of gas used to the gas limit. The predicted variable <code>$Y$</code> represents the normalized gas used, determined by the formula:</p>

<blockquote>
    <p>
        \[ Y = \frac{{\text{{gasUsed}} - \text{{gasTarget}}}}{{\text{{gasTarget}}}} \]
    </p>
</blockquote>

<p>For varying periods <code>$k$</code>, the regressor variable for the preceding <code>$k$</code> data points is collected into a list, forming the feature set <code>$X$</code>. The variable <code>$Y$</code> corresponds precisely to the prediction variable for the data point at time <code>$t$</code>.</p>

#### Off-chain data
We focused our analysis on key communities within the blockchain ecosystem, specifically targeting the Binance, Uniswap, and Ethereum Dev channels on Discord. These communities represent major segments of the industry: Binance as the largest centralized exchange, Uniswap as a leading decentralized exchange, and Ethereum Dev as a hub for Ethereum developers. Our data collection involved querying the discussion text from the "general" channels of these three communities.We query all the discussion data of the selected channels, starting from the establish date of each channel until April 2, 2024. After aggregating the text data, we compiled a total of 728863 individual discussion entries. 
<table>
        <thead>
            <tr>
                <th>AuthorID</th>
                <th>Date</th>
                <th>Content</th>
                <th>Attach</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>301186049323958272</td>
                <td>2019-08-24 13:47</td>
                <td>If you believe that each of these stable coins are eventually stable though, then you'll make money on the swinging back and forth.</td>
                <td>NA</td>
            </tr>
            <tr>
                <td>510252034310799360</td>
                <td>2019-08-24 14:59</td>
                <td>Heh, I was imagining a world where the pool has more liquidity than the rest of the market participants combined. Your, actually realistic, scenario makes it a little difficult for pool participants to exit.</td>
                <td>NA</td>
            </tr>
            <tr>
                <td>589621262733672448</td>
                <td>2019-08-24 17:02</td>
                <td>I kinda prefer variable price.</td>
                <td>NA</td>
            </tr>
        </tbody>
    </table>

#### Data processing 
we also incorporate an additional off-chain data source, specifically the discussion text from Discord. To analyze this data, we use a large language model to process English sentences or words, estimating the probability of each sentence being classified as positive, negative, or neutral, ensuring that the total probability sums to 1. After obtaining sentiment information, we organize the corpus sequentially and compute average sentiment scores over both hourly and daily intervals. This sentiment information is denoted as gammma. We then synchronize the on-chain data with the off-chain sentiment using corresponding block data from the previous time chunk, ensuring that only preceding sentiment information is included in the training data.


### Monotonicity
we propose a novel method for predicting gas usage in blockchain transactions, inspired by the concept of pairwise monotonicity as detailed by Chen \cite{chen2023address}. Unlike traditional methods like EMA, which emphasizes the forgetting of older information, our approach employs a monotonicity representation to attribute varying levels of importance to data over time. Monotonicity has demonstrated its interdisciplinary applicability, as evidenced by works such as Liu et al. \cite{liu2020certified} and Milani \cite{milani2016fast}, which focused on individual monotonicity for single variables. Our method is inspired by Chen's work \cite{chen2023address} for introducing pairwise monotonicity in the financial domain. For instance, past due amounts over a longer period in credit scoring should more significantly impact the scoring of new debt risk. Similarly, older data points are less influential in blockchain transactions, whereas recent data points are more critical for prediction.

We apply monotonicity to the α feature, where changes in α for recent blocks result in greater prediction variance than changes in distant previous data points. 

### FinBert model
The sentiment extraction from the data is conducted using FinBert, a model proposed by Dogu and Araci in 2019 \cite{araci2019finbert}. FinBert is a BERT-based architecture specifically trained on financial data sets, including the Financial PhraseBank, TRC2-financial, and FiQA Sentiment. This training enables FinBert to achieve state-of-the-art performance in FiQA sentiment scoring. Our research uses FinBert to predict sentiment in our text data, ensuring accurate sentiment analysis aligned with financial contexts.

### NAM model
The Neural Additive Model (NAM), proposed by Agarwal et al. in 2021 \cite{agarwal2021neural}, offers a transparent framework for utilizing Deep Neural Networks (DNNs) to model individual or combined features. In this architecture, the outputs from all DNNs are aggregated at the final hidden layer, resulting in a unified model. Our research omits interactions between unrelated features, enabling the imposition of weak monotonicity constraints on each feature.

