# Random Fourier Feature Kernel Recursive Least Squares
## Abstract
In this paper, we investigate the nonlinear, finite dimensional and data independent random Fourier feature expansions that can approximate the popular Gaussian kernel. With recursive least squares algorithm, we develop the Random Fourier Feature Recursive Least Squares algorithm (RFF-RLS), which shows significant performance improvements in simula-tions when compared with several other online kernel learning algorithms such as Kernel Least Mean Square (KLMS) and Kerne Recursive Least Squares (KRLS). Our results confirm that the RFF-RLS can achieve desirable performance with low computational cost. As for the random Fourier features, the randomization generally results in redundancy. We use an algorithm, namely, Vector Quantization with Information Theoretic Learning (VQIT) to decrease the dictionary size. The resulting sparse dictionary can match the original data distribution well. The RFF-RLS with VQIT can outperform the RFF-RLS without VQIT.
## Correction of the error in the article
Change <a href="https://www.codecogs.com/eqnedit.php?latex=$&space;\textbf{P}\left(i&plus;1\right)=\textbf{P}\left(i\right)\textbf{k}\textbf{s}^{T}&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$&space;\textbf{P}\left(i&plus;1\right)=\textbf{P}\left(i\right)\textbf{k}\textbf{s}^{T}&space;$" title="$ \textbf{P}\left(i+1\right)=\textbf{P}\left(i\right)\textbf{k}\textbf{s}^{T} $" /></a>
to <a href="https://www.codecogs.com/eqnedit.php?latex=$&space;\textbf{P}\left(i&plus;1\right)={\beta^{-1}}(\textbf{P}\left(i\right)-\textbf{k}\textbf{s}^{T})&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$&space;\textbf{P}\left(i&plus;1\right)={\beta^{-1}}(\textbf{P}\left(i\right)-\textbf{k}\textbf{s}^{T})&space;$" title="$ \textbf{P}\left(i+1\right)={\beta^{-1}}(\textbf{P}\left(i\right)-\textbf{k}\textbf{s}^{T}) $" /></a>
in the discription of Algorithm 1 in page 3. 
This formula is one of the iterative formulas in the RLS algorithm, but it was written incorrectly in the article. 
Therefore, in this document we correct the wrong formula and published the code of the experiment.
## Language
Matlab
## Cite
If you use this code, please cite the following paper:

[1]	Z. Qin, B. Chen, and N. Zheng, “Random fourier feature kernel recursive least squares,” in Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017, pp. 2881–2886.
