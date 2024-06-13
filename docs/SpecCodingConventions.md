# Coding Conventions

This document captures the WebNN spec repo conventions. The intended audience is editors and contributors to the specification.

When updating these conventions, consider whether an automated check can be added to the [lint tool](../tools/lint.mjs) to help catch issues.

## Resources

* [Writing Procedural Specs](https://garykac.github.io/procspec/)
* [Writing Specifications with Bikeshed](https://dlaliberte.github.io/bikeshed-intro/)
* [Bikeshed Docs](https://speced.github.io/bikeshed/)

## Background Material

The [Web IDL Standard](https://webidl.spec.whatwg.org/) standard defines the interface between JavaScript and an implementation of the standard. Web IDL is the interface between your spec and JavaScript. 99.99% of the time your spec shouldn't mention anything related to JavaScript outside of examples. (So nothing Normative). And that 0.01% is REALLY HARD to get right, leads to security issues, etc.

The [Infra](https://infra.spec.whatwg.org/) standard captures the fundamental concepts on which specifications are built - algorithms, primitive data types, and structures. Most Normative spec text should be written in terms of these concepts. That said, Infra is not perfect or complete. If you need to go beyond it, look at other specs for inspiration - it's better to follow precedent. Look at open Infra issues and consider filing issues if you're in novel territory.


### High Level Spec Authoring Guidance

* Earlier web specs gave high level description of behavior, e.g. input and output. But it turns out a lot of the implementation can be observable and thus lead to compatibility problems. Developers don't test everywhere or read warnings. Specifications must be complete and precise, and not leave room for interpretation.
* Specs are for a specific audience: implementers; if you try to write for developers, you're gonna have a bad time. They should be looking at MDN.
* If you define anything in two places you have twice the chances of getting it wrong. If you get out of sync, everything is wrong. Prioritize precision over readability.
* That said, non-normative notes are your friend. So-called "domintro" sections help reconcile a lot of the tensions above. They're developer-facing non-normative notes in the spec. Use simpler phrasing, don't worry as much about edge cases, etc.


### Bikeshed Guidance

Think of Bikeshed as a compiler for your spec.

* Like code: the less you write the lower your chances of having a bug.
* Prefer markdown to markup - shorter, more likely to catch errors.
* Use links to terms as often as possible. This has many benefits:
    * It handles formatting automatically and reduces manual work.
    * It removes ambiguity about what you're describing.
    * It catches errors if you spell something incorrectly or the term changes.
    * If there is nothing to link to, it is a signal that your text is not precise!
* The spec build should be warning-free and error-free. ("--die-on=warning")


## Specifics

### Definitions And Linking

* Prefer Bikeshed [autolink shortcuts](https://speced.github.io/bikeshed/#autolink-shortcuts) where possible using markdown. These do basic type checking and will style the link appropriately. The most common forms are:
    * `[=term=]` to link to a term
    * `[=scope/term=]` to link to a scoped term (see below)
    * `{{Type}}` to link to a type defined in Web IDL
    * `{{Type/property}}` to link to a member defined in Web IDL
    * `[[doc]]` to link to another document.
* Do not use HTML links `<a href="...">...</a>"` except as a last resort - there is usually a way to create the link using Bikeshed and some error checking.
* Where possible, prefer auto-generated IDs for terms over explicit IDs (`data-lt`).

Example:
```
    The <dfn>maximum size</dfn> of a collection is...

    1. Let |max| be |collection|'s [=maximum size=].
```

* When defining subsidiary terms, like properties of an object, members of an enum, etc, scope the definitions using `dfn-for` on the `dfn` or an ancestor.

Example:
```
    A <dfn>circle</dfn> is a geometric shape.

    <div dfn-for="circle">
    A [=circle=] has an <dfn>origin</dfn> and a <dfn>radius</dfn>.
    </div>

    1. If |shape| is a [=circle=], draw it at |shape|'s [=circle/origin=].
```

* When referencing an operator in text (e.g. sigmoid, tanh, etc), link the operator name to the `MLGraphBuilder` methods for creating the corresponding `MLOperand` or `MLActivation`, e.g. `{{MLGraphBuilder/sigmoid()}}`. This provides consistent styling, and provides a thorough overview of the operator, even if the method itself isn't being discussed.


### Characters and Encoding

* The spec is encoded with UTF-8.
* For non-ASCII characters, prefer to use characters directly, rather than [character references](https://html.spec.whatwg.org/multipage/syntax.html#character-references) (a.k.a. entities), except when necessary for escaping e.g. `sequence&lt;DOMString&gt;`. These commonly occur in names in the Acknowledgements and References sections.
* Commonly used punctuation and symbol characters include:
    * « » (U+00AB / U+00BB Left/Right Pointing Double Angle Quotation Marks) used for [list literals](https://infra.spec.whatwg.org/#lists) and [map literals](https://infra.spec.whatwg.org/#maps).
    * → (U+2192 Rightwards Arrow) used for [map iteration](https://infra.spec.whatwg.org/#map-iterate) and [map literals](https://infra.spec.whatwg.org/#maps).
* In expressions:
    * Use * (U+002A Asterisk) for multiplication, / (U+002F Solidus) for division, and - (U+002D Hyphen-Minux), to reduce friction for implementers. Don't use × (U+00D7 Multiplication Sign), ∗ (U+2217 Asterisk Operator), ÷ (U+00F7 Division Sign), or − (U+2212 Minus Sign).
    * Use named functions like _floor(x)_ and _ceil()_ rather than syntax like ⌊_x_⌋ and ⌈_x_⌉.


### Formatting

* Bikeshed will automatically style linked terms appropriately, for example Web IDL types show up as `code`. Try to avoid manual styling wherever possible; if you're not getting the style you expect, you may have incorrect definitions or links.
* Outside of examples, which should be appropriately styled automatically, literals such as numbers within spec prose are not JavaScript values and should not be styled as code.
* Strings used internally (e.g. operator names) should not be styled as code.
* When concisely defining a list's members or a tensor's layout, use the syntax `*[ ... ]*` (e.g. _"nchw" means the input tensor has the layout *[batches, inputChannels, height, width]*_)
* In Web IDL `<pre class=idl>` blocks, wrap long lines to avoid horizontal scrollbars. 88 characters seems to be the magic number.


### Algorithms

* Use `<div algorithm>` or a variant to wrap the algorithm to get special formatting and Bikeshed processing.
* Number all steps with `1.` which results in automatically incrementing numbers. This makes insertions and deletions much easier.
* Use `|pipe|` syntax for arguments and variables. Bikeshed will give warnings if a variable is only defined once in an algorithm.
* Use assertions when state within an algorithm may not be clear.
* Do not use assertions to repeat what is stated in an algorithm's declaration, e.g. types.
* Do not include conditions that can never be true. For example, if an internal algorithm will only be called with objects in a known state, do not include checks for that state that alter the flow of the algorithm. Consider assertions instead.
* Use the most specific types possible (e.g. MLOperand, not generic object).
* Use `[=this=]` to refer to the current object.
* Use `[=map/For each=] |key| → |value| of |map|` when iterating over a map, but use more specific terms for the key and value (e.g. _For each name → input of inputs:_)
* Use `[=list/For each=] |item| of |list|` when iterating over a list, but use more specific terms for the item (e.g. _For each dimension of dimensions:_)
* Use `[=list/For each=] |index| in [=the range=] X to Y, inclusive` when iterating over a numeric range; a range is implicitly an ordered set which is a type of list. Specify _inclusive_ or _exclusive_ regarding the upper bound, for clarity.
* Use "let" to introduce a variable and "set" to update a variable or assign to a property.
* Use « » notation for literal [lists](https://infra.spec.whatwg.org/#lists), which helps make it clear that they are not JavaScript arrays.
* Use «[ _k_ → _v_ ]» notation for literal [maps](https://infra.spec.whatwg.org/#maps).
* When referring to abstract properties, use the short possessive form `|object|'s [=property=]`. Avoid the wordier `the [=property=] of |object|` form.
* Use "rank" when describing the number of dimensions of a tensor (e.g. in variable names) rather than the ambiguous "size".
* Only use single capital letters as variable names when referring to tensors; i.e. prefer `|shapeA|` to `|A|`, but tensor `|T|` is okay.


### Method Definitions

* Follow the Web IDL convention for [defining methods](https://webidl.spec.whatwg.org/#method-steps), i.e. _"The operation(arg1, arg2, ...) method steps are:"_
* The definition should be wrapped in `<dfn>` and if written correctly will link to the Web IDL declaration.
* Do not include assertions about argument types. This is redundant with Web IDL declaration.
* Do not include steps that test argument types if those types are guaranteed by WebIDL.
* Do not refer to JavaScript or WebIDL types in method steps. Per the spec processing model, by the time a spec algorithm is invoked, JavaScript types (e.g. Numbers, Arrays) have been mapped to WebIDL types (e.g. unsigned longs, sequences) and those have been mapped to Infra types or general concepts (e.g. numbers, lists).
* Do not repeat detaults provided by the WebIDL declaration.
* For types like lists that can't be defaulted in WebIDL, define the default when missing as an explicit step. Example: _If options.padding does not exist, set options.padding to « 0, 0, 0, 0 »._


### Internal Algorithms

* Follow the Infra convention for [algorithm definition](https://infra.spec.whatwg.org/#algorithm-declaration), i.e. _"To [algorithm name], given a [type1] [parameter1], a [type2] [parameter2], …, perform the following steps. They return a [return type]."_
* Bikeshed is smart enough to link verb phrases to algorithms. For example, if you define an algorithm `To <dfn>screw in a lightbulb</dfn> ...` you can link to it with the very readable `... after [=screwing in a lightbulb=] ...`; explicit link targets and alias text is not necessary. Use this form when possible, rather than the explicit "invoking" term.


### Exceptions

* Exceptions are thrown with the syntax: `[=exception/throw=] a {{TypeError}}` or `[=exception/throw=] an "{{OperationError}}" {{DOMException}}`


### Dictionary Members

* Dictionary members are referenced using dotted property syntax. e.g. _options.padding_
   * Note that this is contrary to Web IDL + Infra; formally, a JavaScript object has been mapped to a Web IDL [dictionary](https://webidl.spec.whatwg.org/#idl-dictionaries) and then processed into an Infra [map](ordered) by the time a spec is using it. So formally the syntax _options["padding"]_ should be used.
* Dictionary members should be given definitions somewhere in the text. This is usually done with a `<dl dfn-type=dict-member dfn-for=...>` for the dictionary as a whole, containing a `<dfn>` for each member.
