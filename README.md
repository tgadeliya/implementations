# 🫖 rooibos

>  /ˈrɔɪbɒs/ ROY-boss; Afrikaans: [ˈroːibɔs], lit. 'red bush'

A bunch of ML and DL algorithm implementations with tests. I kept most of them simple so they can be quickly reimplemented from scratch.
I started this repo after failing a CodeSignal assignment — I was so shocked they asked me to implement things like dot product in plain Python that I totally froze

#### Features
- ml algorithms implemented in python and numpy
- dl algorithms implementation in numpy, torch, jax(maybe)
- test suits for every algorithm (WIP)
- notebooks to train implementations and additional tests to better understand algorithms (coming soon)

## Algorithms
| ML |
| --- |
| [`Linear Regression`](https://github.com/tgadeliya/implementations/tree/main/src/rooibos/ml/linear_regression) |
| [`Logistic Regression`](https://github.com/tgadeliya/implementations/tree/main/src/rooibos/ml/logistic_regression) |
| [`K-Means`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/ml/clustering/k_means.py) |
| [`KNN`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/ml/nearest_neighbours/knn.py) |
| [`Decision Tree`](https://github.com/tgadeliya/implementations/tree/main/src/rooibos/ml/tree) |

| DL |
| --- |
| [`BPE tokenizer and trainer`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/bpe_tokenizer.py) |
| [`Transformer for Language Modeling`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/models.py) |
| [`Multi-head Attention`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py) |
| [`Rotary Position Embeddings (RoPE)`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py) |
| Other transformer components: [`RMSNorm`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py),[`Embedding`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py), [`Linear`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py), [`SiLU`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py), [`SwiGLU`](https://github.com/tgadeliya/implementations/blob/main/src/rooibos/dl/transformer/components.py) |


## References

- [`Stanford CS336`](https://stanford-cs336.github.io/spring2025/)
- [`mlcourse.ai`](https://mlcourse.ai/book/index.html)
- Kevin P. Murphy [`"Probabilistic Machine Learning: An Introduction"`](https://probml.github.io/pml-book/book1.html)
- Kevin P. Murphy [`"Probabilistic Machine Learning: Advanced Topics"`](https://probml.github.io/pml-book/book2.html)
