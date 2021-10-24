# mcdm

Python implementation of multiple-criteria decision-making algorithms

<!-- START OF BADGES -->
![Status of tests workflow](https://img.shields.io/github/workflow/status/akestoridis/mcdm/wf01-tests?label=tests)
![Status of coverage workflow](https://img.shields.io/github/workflow/status/akestoridis/mcdm/wf02-coverage?label=coverage)
![Status of quality workflow](https://img.shields.io/github/workflow/status/akestoridis/mcdm/wf03-quality?label=quality)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/akestoridis/mcdm)
![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/akestoridis/mcdm/latest)
![Python version requirement](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
<!-- END OF BADGES -->


## Installation

The `mcdm` package can be installed from PyPI using pip for Python 3:
```console
$ pip3 install mcdm
```

Alternatively, you can install the latest version of the `mcdm` package from its GitHub repository:
```console
$ git clone https://github.com/akestoridis/mcdm.git
$ cd mcdm/
$ pip3 install .
```


## Features

The following tables include the scoring, weighting, correlation, and normalization methods that are supported by the `mcdm` package.

### Scoring methods
| Short Name | Full Name                                                               | References                 |
| ---------- | ----------------------------------------------------------------------- | -------------------------- |
| SAW        | Simple Additive Weighting                                               | [[1]](#ref1), [[2]](#ref2) |
| MEW        | Multiplicative Exponential Weighting                                    | [[2]](#ref2)               |
| TOPSIS     | Technique for Order Preference by Similarity to Ideal Solution          | [[1]](#ref1)               |
| mTOPSIS    | Modified Technique for Order Preference by Similarity to Ideal Solution | [[3]](#ref3)               |

### Weighting methods
| Short Name | Full Name                                             | References                 |
| ---------- | ----------------------------------------------------- | -------------------------- |
| MW         | Mean Weights                                          | [[4]](#ref4)               |
| EM         | Entropy Measure                                       | [[1]](#ref1), [[3]](#ref3) |
| SD         | Standard Deviation                                    | [[4]](#ref4)               |
| CRITIC     | Criteria Importance Through Intercriteria Correlation | [[4]](#ref4)               |
| VIC        | Variability and Interdependencies of Criteria         | [[5]](#ref5)               |

### Correlation methods
| Short Name | Full Name                                              | References                 |
| ---------- | ------------------------------------------------------ | -------------------------- |
| Pearson    | Pearson Correlation Coefficients                       | [[6]](#ref6)               |
| AbsPearson | Absolute Value of the Pearson Correlation Coefficients | [[6]](#ref6)               |
| dCor       | Distance Correlation Coefficients                      | [[7]](#ref7), [[8]](#ref8) |

### Normalization methods
| Short Name | Full Name                | References                 |
| ---------- | ------------------------ | -------------------------- |
| Linear1    | Linear Normalization (1) | [[1]](#ref1), [[9]](#ref9) |
| Linear2    | Linear Normalization (2) | [[1]](#ref1), [[9]](#ref9) |
| Linear3    | Linear Normalization (3) | [[1]](#ref1), [[9]](#ref9) |
| Vector     | Vector Normalization     | [[1]](#ref1), [[9]](#ref9) |


## Usage

After importing the `mcdm` package, you can view its contents using the built-in `help` function:
```pycon
>>> import mcdm
>>> help(mcdm)
```

The contents of its subpackages can be viewed similarly, e.g.:
```pycon
>>> help(mcdm.weighting)
```

The `mcdm` package can compute the ranking of alternatives, which are provided as an `array_like` object, with its `rank` function.
By default, the `rank` function is using the SAW scoring method, the MW weighting method, and assumes that the decision matrix contains unnamed alternatives with normalized benefit criteria:
```pycon
>>> x_matrix = [
...     [0.00, 1.00],
...     [0.25, 0.75],
...     [0.50, 0.50],
...     [0.75, 0.25],
...     [1.00, 0.00],
... ]
>>> mcdm.rank(x_matrix)
[('a1', 0.5), ('a2', 0.5), ('a3', 0.5), ('a4', 0.5), ('a5', 0.5)]
```

You can select the use of the MEW scoring method, without changing the remaining default selections, as follows:
```pycon
>>> x_matrix = [
...     [0.00, 1.00],
...     [0.25, 0.75],
...     [0.50, 0.50],
...     [0.75, 0.25],
...     [1.00, 0.00],
... ]
>>> mcdm.rank(x_matrix, s_method="MEW")
[('a3', 0.5000000000000001), ('a2', 0.4330127018922193), ('a4', 0.4330127018922193), ('a1', 0.0), ('a5', 0.0)]
```

Alternatively, you can use the TOPSIS scoring method with predefined weights as follows:
```pycon
>>> x_matrix = [
...     [0.00, 1.00],
...     [0.25, 0.75],
...     [0.50, 0.50],
...     [0.75, 0.25],
...     [1.00, 0.00],
... ]
>>> mcdm.rank(x_matrix, w_vector=[0.7, 0.3], s_method="TOPSIS")
[('a5', 0.7), ('a4', 0.6504133360970108), ('a3', 0.5), ('a2', 0.3495866639029891), ('a1', 0.3)]
```

You can also use the TOPSIS scoring method with a mixture of benefit and cost criteria as follows:
```pycon
>>> x_matrix = [
...     [0.00, 1.00],
...     [0.25, 0.75],
...     [0.50, 0.50],
...     [0.75, 0.25],
...     [1.00, 0.00],
... ]
>>> mcdm.rank(x_matrix, is_benefit_x=[True, False], s_method="TOPSIS")
[('a5', 1.0), ('a4', 0.75), ('a3', 0.5), ('a2', 0.25000000000000006), ('a1', 0.0)]
```

Alternatively, you can use the TOPSIS scoring method, the SD weighting method, and the Vector normalization method with named alternatives as follows:
```pycon
>>> x_matrix = [
...     [4,  5, 10],
...     [3, 10,  6],
...     [3, 20,  2],
...     [2, 15,  5],
... ]
>>> alt_names = ["A", "B", "C", "D"]
>>> mcdm.rank(x_matrix, alt_names=alt_names, n_method="Vector", w_method="SD", s_method="TOPSIS")
[('A', 0.5623140105790617), ('D', 0.472563994792934), ('C', 0.4474283120076966), ('B', 0.43874437587505694)]
```

Similarly, you can use the SAW scoring method, the CRITIC weighting method, and the Linear2 normalization method with named alternatives as follows:
```pycon
>>> x_matrix = [
...     [4,  5, 10],
...     [3, 10,  6],
...     [3, 20,  2],
...     [2, 15,  5],
... ]
>>> alt_names = ["A", "B", "C", "D"]
>>> mcdm.rank(x_matrix, alt_names=alt_names, n_method="Linear2", w_method="CRITIC", s_method="SAW")
[('C', 0.5864039798997854), ('A', 0.5363555775174913), ('B', 0.42272592958624855), ('D', 0.41815995516110754)]
```

Furthermore, you can use the mTOPSIS scoring method, the EM weighting method, and the Linear3 normalization method with named alternatives as follows:
```pycon
>>> x_matrix = [
...     [4,  5, 10],
...     [3, 10,  6],
...     [3, 20,  2],
...     [2, 15,  5],
... ]
>>> alt_names = ["A", "B", "C", "D"]
>>> mcdm.rank(x_matrix, alt_names=alt_names, n_method="Linear3", w_method="EM", s_method="mTOPSIS")
[('A', 0.5671982017516887), ('D', 0.4737709007480381), ('B', 0.44023602515388915), ('C', 0.43979056725587967)]
```

In addition, you can use the MEW scoring method, the VIC weighting method, and the Linear1 normalization method with named alternatives as follows:
```pycon
>>> x_matrix = [
...     [4,  5, 10],
...     [3, 10,  6],
...     [3, 20,  2],
...     [2, 15,  5],
... ]
>>> alt_names = ["A", "B", "C", "D"]
>>> mcdm.rank(x_matrix, alt_names=alt_names, n_method="Linear1", w_method="VIC", s_method="MEW")
[('A', 0.596199006150288), ('B', 0.5926510141687035), ('D', 0.5816528401371021), ('C', 0.507066254464828)]
```

Finally, you can use the `load` function of the `mcdm` package to load a decision matrix from a text file (e.g., the [example09.tsv](https://github.com/akestoridis/mcdm/raw/77d526b93f70eabbe91dc20a88aa1347459e4e75/mcdm/tests/data/example09.tsv) file), and then compute the ranking of its alternatives using the MEW scoring method and the VIC weighting method as follows:
```pycon
>>> x_matrix, alt_names = mcdm.load("./mcdm/tests/data/example09.tsv", delimiter="\t", skiprows=1, labeled_rows=True)
>>> mcdm.rank(x_matrix, alt_names=alt_names, w_method="VIC", s_method="MEW")
[('COORD.PRoPHET', 0.47540101629920883), ('DF.PRoPHET', 0.4720540449389032), ('CnR.LTS', 0.38076976314696165), ('SimBetTS.L8', 0.3800058193419937), ('SimBetTS.L16', 0.3799920328578032), ('CnR.DestEnc', 0.37944808013507936), ('LSF-SnW.L16', 0.37739981242275067), ('DF.DestEnc', 0.3737879965369727), ('COORD.DestEnc', 0.3735362169300779), ('SimBetTS.L4', 0.372439515643607), ('LSF-SnW.L8', 0.3689450285406012), ('DF.LTS', 0.36604297140966213), ('COORD.LTS', 0.36532018876831296), ('LSF-SnW.L4', 0.34498575401083065), ('CnF.PRoPHET', 0.344899433667112), ('CnF.DestEnc', 0.34080904510687654), ('CnF.LTS', 0.33682425293123014), ('SnF.L8', 0.3338134560941729), ('SnF.L4', 0.3310799577048607), ('CnR.PRoPHET', 0.3283706628162786), ('SnF.L2', 0.3282710142810222), ('SnF.L16', 0.325965295985982), ('SimBetTS.L2', 0.3198197170434966), ('LSF-SnW.L2', 0.28336307866897725), ('CnR.Enc', 0.25388909503755097), ('DF.Enc', 0.19642752820544426), ('COORD.Enc', 0.18527125018989776), ('Epidemic', 0.17618218317052287), ('Direct', 0.14463684900589485), ('EBR.L16', 0.14427544773753895), ('SnW.L16', 0.14419569083973272), ('EBR.L2', 0.139576851541699), ('SnW.L2', 0.1393465080643217), ('SnW.L8', 0.13728835719879856), ('EBR.L8', 0.13728300706136987), ('EBR.L4', 0.13654721879934206), ('SnW.L4', 0.1364251455180083), ('CnF.Enc', 0.11713353969310777)]
```


## References

<a name="ref1">**[1]**</a> C.-L. Hwang and K. Yoon, *Multiple Attribute Decision Making*, ser. Lecture Notes in Economics and Mathematical Systems. Springer-Verlag Berlin Heidelberg, 1981, vol. 186, isbn: 9783540105589.

<a name="ref2">**[2]**</a> S. H. Zanakis, A. Solomon, N. Wishart, and S. Dublish, “Multi-attribute decision making: A simulation comparison of select methods,” *Eur. J. Oper. Res.*, vol. 107, no. 3, pp. 507–529, 1998, doi: [10.1016/S0377-2217(97)00147-1](https://doi.org/10.1016/S0377-2217(97)00147-1).

<a name="ref3">**[3]**</a> H. Deng, C.-H. Yeh, and R. J. Willis, “Inter-company comparison using modified TOPSIS with objective weights,” *Comput. Oper. Res.*, vol. 27, no. 10, pp. 963–973, 2000, doi: [10.1016/S0305-0548(99)00069-6](https://doi.org/10.1016/S0305-0548(99)00069-6).

<a name="ref4">**[4]**</a> D. Diakoulaki, G. Mavrotas, and L. Papayannakis, “Determining objective weights in multiple criteria problems: The CRITIC method,” *Comput. Oper. Res.*, vol. 22, no. 7, pp. 763–770, 1995, doi: [10.1016/0305-0548(94)00059-H](https://doi.org/10.1016/0305-0548(94)00059-H).

<a name="ref5">**[5]**</a> D.-G. Akestoridis and E. Papapetrou, “A framework for the evaluation of routing protocols in opportunistic networks,” *Comput. Commun.*, vol. 145, pp. 14–28, 2019, doi: [10.1016/j.comcom.2019.06.003](https://doi.org/10.1016/j.comcom.2019.06.003).

<a name="ref6">**[6]**</a> J. L. Rodgers and W. A. Nicewander, “Thirteen ways to look at the correlation coefficient,” *Amer. Statist.*, vol. 42, no. 1, pp. 59–66, 1988, doi: [10.2307/2685263](https://doi.org/10.2307/2685263).

<a name="ref7">**[7]**</a> G. J. Székely, M. L. Rizzo, and N. K. Bakirov, “Measuring and testing dependence by correlation of distances,” *Ann. Statist.*, vol. 35, no. 6, pp. 2769–2794, 2007, doi: [10.1214/009053607000000505](https://doi.org/10.1214/009053607000000505).

<a name="ref8">**[8]**</a> G. J. Székely and M. L. Rizzo, “Brownian distance covariance,” *Ann. Appl. Statist.*, vol. 3, no. 4, pp. 1236–1265, 2009, doi: [10.1214/09-AOAS312](https://doi.org/10.1214/09-AOAS312).

<a name="ref9">**[9]**</a> H.-S. Shih, H.-J. Shyur, and E. S. Lee, “An extension of TOPSIS for group decision making,” *Math. Comput. Model.*, vol. 45, no. 7–8, pp. 801–813, 2007, doi: [10.1016/j.mcm.2006.03.023](https://doi.org/10.1016/j.mcm.2006.03.023).


## License

Copyright (c) 2020-2021 Dimitrios-Georgios Akestoridis

This project is licensed under the terms of the MIT License (MIT).
