# Previous results and possible research questions

## Syntactic correctness of generated formulae
In our previous work, we observed that the STLDecoder architecture (a MarianMT-inspired decoder-only structure) is able to recostruct syntactically-valid formulae starting from their kernel embeddings almost completely after 3 epochs.
For certain training sets configurations (i.e. `random` and `random-small`) this phenomenon happened earlier, after almost one epoch.
On the opposite, `balanced`-trained models seem terrible and also `easyskewed` and `hardskewed` do not seem able to do so after such a little number of steps (<50% of syntactically correctness achieved). 

- What happens in model's internals such that almost all-decoded formulae become valid in 3 epochs?
- Why do we observe such difference between different training sets in terms of validity after a single epoch?
- Which is the hidden-dimension impact with respect to formulae syntactic correctness?
- Why do some epochs cause an (apparent) decrease in percentage of syntactic valid formulae with respect to previous steps?

## Semantic meaning fidelity
We compared $\text{cos}(\rho(\phi), \rho(\hat{\phi}))$ and $\text{diff}(s(\phi), s(\hat{\phi}))$ of the pairs of gold ($\phi$) and generated ($\hat{\phi}$) formulae respectively.
Note that: 
- $\rho(\phi)$ is the robustness vector associated to the formula $\phi$ (i.e. a proxy of the embedding) and is $M$-dimensional (where $M$ is a certain number of trajectories);
- the cosine similarity between the two robustness vectors of the gold ($\phi$) and the generated ($\hat{\phi}$) formulae measures how similar these two formulae are in absolute terms;
- $s(\phi)$ is the sign of whether the formula $\phi$ is satisfied on a certain set of trajectories;
- $\text{diff}(s(\phi), s(\hat{\phi})) = 0$ if the original formula $\phi$ and the recostructed one $\hat{\phi}$ are satisfied or unsatisfied on the same set of trajectories.


