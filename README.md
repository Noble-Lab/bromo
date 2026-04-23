# bromo

bromo is an deep learning model for prioritizing peptides in targeted mass spectrometry experiments. The associated preprint "Prioritizing peptides for targeted mass spectrometry experiments using deep learning" is available here.
bromo ranks peptide precursors within a protein by their predicted MS2 response in DIA experiments using only amino acid sequence and charge state as input. Unlike existing tools that rely on detectability as a proxy or small synthetic training sets, bromo is trained on millions of peptide pairs derived from large-scale, publicly available DIA data and consistently outperforms existing sequence-based methods across diverse, independent datasets. bromo can also be fine-tuned on experiment-specific data to account for differences in sample preparation, sample matrix, and instrument platform. For reproducing the analyses in the manuscript, please visit our manuscript repo.
bromo is open source under an Apache 2.0 license.

<p align="center">
  <img src="docs/images/bromo_icon.png" width="400" height="400">
</p>