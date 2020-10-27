[![build](https://github.com/webmachinelearning/webnn/workflows/build/badge.svg)](https://github.com/webmachinelearning/webnn/actions)

Web Neural Network API
=======

This repository hosts the [Web Neural Network API](https://webmachinelearning.github.io/webnn/)
being worked on in the 
[Machine Learning for the Web Community Group](https://www.w3.org/community/webmachinelearning/).

## Generating the specification

The specification is written using [Bikeshed](https://tabatkins.github.io/bikeshed).

If you have bikeshed installed locally, you can generate the specification with:

```
prompt> make
```

This simply runs bikeshed on the `index.bs` file.

Otherwise, you can use the bikeshed Web API:

```
prompt> make online=1
```
