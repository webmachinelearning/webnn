[![build](https://github.com/webmachinelearning/webnn/workflows/build/badge.svg)](https://github.com/webmachinelearning/webnn/actions)

Web Neural Network API
=======

This repository hosts the [Web Neural Network API](https://webmachinelearning.github.io/webnn/)
being worked on in the 
[Web Machine Learning Working Group](https://www.w3.org/groups/wg/webmachinelearning).

## Pull requests

To propose a change, update `index.bs` and submit your PR. The `index.html` file is automatically built and deployed upon merge.

The spec is written using [Bikeshed](https://tabatkins.github.io/bikeshed). Please refer to the [Bikeshed Documentation](https://tabatkins.github.io/bikeshed/) for the Bikeshed syntax.

For testing purposes, you can generate the spec locally:

If you have bikeshed [installed locally](https://tabatkins.github.io/bikeshed/#installing), you can generate the specification locally with:

```
prompt> make
```

This simply runs bikeshed on the `index.bs` file.

Otherwise, you can use the [bikeshed Web API](https://tabatkins.github.io/bikeshed/#remote):

```
prompt> make online=1
```
