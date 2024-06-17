# Assera Compilers

## Assera

[Assera](https://pypi.org/project/assera) is a programming model, a domain-specific programming language embedded in Python (eDSL), and an optimizing cross-compiler for compute-intensive code. Assera currently supports CPU and GPU targets and focuses on optimization of nested for-loops.

Writing highly optimized compute-intensive code in a traditional programming language is a difficult and time-consuming process. It requires special engineering skills, such as fluency in Assembly language and a deep understanding of computer architecture. Manually optimizing the simplest numerical algorithms already requires a significant engineering effort. Moreover, highly optimized numerical code is prone to bugs, is often hard to read and maintain, and needs to be reimplemented every time a new target architecture is introduced. Assera aims to solve these problems.

Assera has three goals:

* Performance: generate the fastest implementation of any compute-intensive algorithm.
* Readability: do so without sacrificing code readability and maintainability.
* Writability: a user-friendly programming model, designed for agility.

## assera-compilers

The `assera-compilers` package contains pre-compiled compiler binaries used to produce optimized code using the Assera eDSL. It is not designed for standalone use, but is automatically installed when you `pip install assera`. You can find documentation and examples on [Github](https://github.com/khulnasoft/Assera).
