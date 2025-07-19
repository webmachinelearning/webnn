#!/usr/bin/env node

// Requires Node.js; to install dependencies, run these steps:
//   cd tools
//   npm install
//   cd ..
//
// Run this script from top level of spec repo after building the spec:
//   bikeshed --die-on=warning spec
//   node tools/lint.mjs
//
// Note that the '.mjs' extension is necessary for Node.js to treat the file as
// a module. There is an `--experimental-default-type=module` flag but
// specifying that in the #! line requires trickery that confuses some editors.
//
// Options:
//   --verbose                    Log progress.
//   --bikeshed PATH_TO_BS_FILE   Bikeshed source path. (default: "index.bs")
//   --html PATH_TO_HTML_FILE     Generated HTML path. (default: "index.html")

// --------------------------------------------------
// About this file
// --------------------------------------------------

// This tool slurps in the source and built files, and then runs various
// tests to catch errors and enforce the coding conventions documented at
// ../docs/SpecCodingConventions.md

// The individual tests are stand-alone and generally have these forms:
// * Run regular expressions over source text.
// * Run a CSS selector over the spec to return a list of matching elements,
//   then run regular expressions over the element text.
// * Run a CSS selector over the spec to return a list of matching elements,
//   then run bespoke checks on the elements using DOM APIs.
//
// Therefore, a good working knowledge of regular expressions, CSS selectors,
// and the DOM API is needed. Additionally, many of the checks are
// WebNN-specific - these are called out. Note that this tool also makes
// many assumptions about the HTML Bikeshed generates, so it will need to
// be updated if Bikeshed makes significant changes.

// --------------------------------------------------
// Dependencies
// --------------------------------------------------

'use strict';
import fs from 'node:fs/promises';
import {parse} from 'node-html-parser';
import * as idl from 'webidl2';

// --------------------------------------------------
// Process command line options
// --------------------------------------------------

const options = {
  verbose: false,
  bikeshed: 'index.bs',
  html: 'index.html',
};

const [interpreter, script, ...args] = globalThis.process.argv;
args.forEach((arg, index, array) => {
  if (arg === '--verbose' || arg === '-v') {
    options.verbose = true;
  } else if (arg === '--bikeshed' && array.length > index + 1) {
    options.bikeshed = array.splice(index + 1, 1)[0];
  } else if (arg === '--html' && array.length > index + 1) {
    options.html = array.splice(index + 1, 1)[0];
  } else {
    console.error(`Unknown or incomplete argument: ${arg}`);
    globalThis.process.exit(1);
  }
});

function log(string) {
  if (options.verbose) {
    console.log(string);
  }
}

// --------------------------------------------------
// Load and parse file
// --------------------------------------------------

// Steps are:
// * Slurp in the Bikeshed source
// * Slurp in the generated HTML
// * Massage the HTML to work around node-html-parser issues
// * Parse the HTML into a DOM using node-html-parser
// * Parse the spec's WebIDL using webidl2

log(`loading Bikeshed source "${options.bikeshed}"...`);
const source = await fs.readFile(options.bikeshed, 'utf8');

log(`loading generated HTML "${options.html}"...`);
let htmlFileData = await fs.readFile(options.html, 'utf8');

log('massaging HTML...');
// node-html-parser doesn't understand that some elements are mutually self-closing;
// tweak the source using regex magic.
// It is perfectly valid in HTML to write `<table><tr><td>...<td>...</table>` to
// produce a table with two cells, since a `TD` can't contain another `TD`
// directly so the second one implicitly closes the first. Bikeshed emits HTML
// like this frequently for several constructs. Since the parse we're stuck with
// here gets confused by this, we massage the HTML to detect these cases and
// insert explicit close tags.
[{tags: ['dt', 'dd'], containers: ['dl']},
 {tags: ['thead', 'tbody', 'tfoot'], containers: ['table']},
 {tags: ['tr'], containers: ['thead', 'tbody', 'tfoot', 'table']},
].forEach(({tags, containers}) => {
  // The regex magic here does the following transformations:
  // Description Lists:
  //  before: `<dl><dt>...<dd>...</dl>`
  //  after:  `<dl><dt>...</dt><dd>...</dd></dl>`
  // Table Sections:
  //  before: `<table><thead>...<tbody>...</table>`
  //  after:  `<table><thead>...</thead><tbody>...</tbody></table>`
  // Table Rows:
  //  before: `<table><tr>...<tr>...</table>`
  //  after:  `<table><tr>...</tr><tr>...</tr></table>`
  const re = new RegExp(
    // For all of the affected open tags (`opener`):
    '(<(' + tags.join('|') + ')\\b[^>]*>)' +
      // Which can contain anything (`content`):
      '(.*?)' +
      // If followed by a similar open tag, or the container close tag:
      '(?=<(' + tags.join('|') + '|/(' + containers.join('|') + '))\\b)',
    'sg');
  htmlFileData = htmlFileData.replaceAll(
    // Then insert the explicit close tag right after the contents.
    re, (_, opener, tag, content) => `${opener}${content}</${tag}>`);
});

log('parsing HTML...');
const root = parse(htmlFileData, {
  blockTextElements: {
    // Explicitly don't list <pre> to force children to be parsed.
    // See https://github.com/taoqf/node-html-parser/issues/78

    // Explicitly list <script> and <style> otherwise `remove()` leaves
    // text content.
    script: true,
    style: true,
  }
});

log('simplifying DOM...');
// Remove script and style elements from consideration. Remove generated indexes
// too, since they can lead to duplicate false-positive matches for lint rules.
for (const element of root.querySelectorAll('script, style, .index')) {
  element.remove();
}

const html = root.innerHTML;
const text = root.innerText;

// Parse the WebIDL Index. This will throw if errors are found, but Bikeshed
// should fail to build the spec if the WebIDL is invalid.
const idl_text = innerText(root.querySelector('#idl-index + pre'));
const idl_ast = idl.parse(idl_text);

// --------------------------------------------------
// Helpers
// --------------------------------------------------

// node-html-parser leaves some entities in innerText; use this helper function
// to translate where it matters, e.g. IDL fragments.
function innerText(element) {
  return element.innerText.replaceAll(/&amp;/g, '&')
    .replaceAll(/&lt;/g, '<')
    .replaceAll(/&gt;/g, '>');
}

// Helper function to format the result of a RegExp match for use in error
// messages. The matched text is presented with a small amount of
// leading/trailing context from the same line.
// e.g. `format(text.match(/÷/))` --> "... numerator ÷ denominator ..."
function format(match) {
  const CONTEXT = 20;

  // Characters in the source string on the same line, before the matched text.
  let prefix = match.input.substring(match.index - CONTEXT, match.index)
                     .split(/\n/)
                     .pop();
  // Characters in the source string on the same line, after the matched text.
  let suffix = match.input.substr(match.index + match[0].length, CONTEXT)
                     .split(/\n/)
                     .shift();
  let infix = match[0];

  // Ignore the prefix/suffix if the matched text starts/ends with a newline,
  // since it could be completely unrelated.
  if (infix.startsWith('\n')) {
    prefix = '';
    infix = infix.slice(1);
  }
  if (infix.endsWith('\n')) {
    suffix = '';
    infix = infix.slice(0, -1);
  }

  return (prefix.length === CONTEXT ? '...' : '') + prefix + infix + suffix +
      (suffix.length === CONTEXT ? '...' : '');
}

// Used later in the script for dynamic script creation. JavaScript has the
// named `Function()` which can be used like `eval()`, but doesn't name the
// async equivalent so we need this one weird trick.
const AsyncFunction = async function() {}.constructor;

// CSS selectors for use with `querySelectorAll()`. These can be combined using
// the usual combinators, e.g. ' ' is the descendant combinator, '>' is the
// child combinator, etc.

// Matches all algorithms in document. Note that this includes algorithms for
// methods and abstract operations, and algorithms may have steps or be compact
// single-line statements.
const ALGORITHM_SELECTOR = '.algorithm';

// Matches all individual algorithm steps in the document. Bikeshed generates LI
// containing OL for sub-steps, and LI containing P for actual steps. Issues
// within steps are ignored.
const ALGORITHM_STEP_SELECTOR = ALGORITHM_SELECTOR + ' li > p:not(.issue)';

// Matches IDL blocks.
const IDL_BLOCK_SELECTOR = 'pre.idl';

// Matches variables.
const VAR_SELECTOR = 'var';

// Matches various definition types.
const ARGUMENT_DFN_SELECTOR = 'dfn[data-dfn-type=argument]';
const DICTMEMBER_DFN_SELECTOR = 'dfn[data-dfn-type=dict-member]';
const INTERFACE_DFN_SELECTOR = 'dfn[data-dfn-type=interface]';
const METHOD_DFN_SELECTOR = 'dfn[data-dfn-type=method]';

// Defined here to make it more obvious why two selectors are being joined by a
// space, for maintainers who don't eat CSS for breakfast.
const DESCENDANT_COMBINATOR = ' ';

// --------------------------------------------------
// Checks
// --------------------------------------------------

let exitCode = 0;

function getLineNumber(htmlFileData, /*HTMLElement*/ node) {
  const result = htmlFileData.substring(0, node.range[0]).match(/\n/g);
  if (result) {
    return result.length + 1;
  }
  return 1;
}

// Failing checks must call `error()` which will log the error message and set
// the process exit code to signal failure.
function error(message) {
  console.error(message);
  exitCode = 1;
}

function errorHtml(message, /*HTMLElement*/ node) {
  console.error(`${options.html}:${getLineNumber(htmlFileData, node)}: ${message}`)
}

log('running checks...');

// Checks can operate on:
// * `source` - raw Bikeshed markdown source
// * `html` - HTML source, with style/script removed
// * `text` - rendered text content
// * `root.querySelectorAll()` - operate on DOM-like nodes
// * `idl_ast` - WebIDL AST (see https://github.com/w3c/webidl2.js)

// Checks are marked with one of these tags:
// * [Generic] - could apply to any spec
// * [WebNN] - very specific to the WebNN spec

// [Generic] Report warnings found when parsing the WebIDL. Bikeshed should
// fail generation on IDL errors, but some warnings may slip through.
for (const err of idl.validate(idl_ast)) {
  error(`WebIDL: ${err.message}`);
}

// [Generic] Look for merge markers
for (const match of source.matchAll(/<{7}|>{7}|^={7}$/mg)) {
  error(`Merge conflict marker: ${format(match)}`);
}

// [Generic] Look for residue of unterminated auto-links in rendered text
for (const match of text.matchAll(/({{|}}|\[=|=\])/g)) {
  error(`Unterminated autolink: ${format(match)}`);
}

// [Generic] Look for duplicate words (in source, since [=realm=] |realm| is okay)
// Note that spaces (or start/end of string) are used as delimiters for the
// regex because there are a lot of potential false positives, including:
// * <a class="p-org org">
// * <code>unsigned long long</code>
// Although this does allow for false negatives. e.g. <p>The the</p>
// TODO: Consider iterating over all text nodes and searching those instead.
for (const match of html.matchAll(/(?:^|\s)(\w+) \1(?:$|\s)/ig)) {
  error(`Duplicate word: ${format(match)}`);
}

// [Generic] Verify IDL lines wrap, to avoid horizontal scrollbars
const MAX_IDL_WIDTH = 88;
for (const idl of root.querySelectorAll(IDL_BLOCK_SELECTOR)) {
  innerText(idl).split(/\n/).forEach(line => {
    if (line.length > MAX_IDL_WIDTH) {
      error(`Overlong IDL: ${line}`);
    }
  });
}

// [WebNN] Look for undesired punctuation
for (const match of text.matchAll(/(::|×|÷|∗|−)/g)) {
  error(`Bad punctuation: ${format(match)}`);
}

// [WebNN] Look for undesired entity usage (e.g. &laquo;, &rarr;, etc)
// Although harder to type, we prefer using «, →, etc. directly since it makes
// the source more readable.
for (const match of source.matchAll(/&(\w+);/g)) {
  if (!['amp', 'lt', 'gt', 'quot'].includes(match[1])) {
    error(`Avoid entities: ${format(match)}`);
  }
}

// [WebNN] Look for undesired phrasing
// e.g. "the [=rank=] of |tensor|" (unless it's the start of a list)
for (const match of source.matchAll(/the (\[=.*?=\]) of (\|.*?\|)[^,]/g)) {
  error(`Prefer "x's y" to "y of x": ${format(match)}`);
}
for (const match of source.matchAll(/1\. Else/ig)) {
  error(`Prefer "otherwise" to "else": ${format(match)}`);
}
for (const match of text.matchAll(/ not the same as /g)) {
  error(`Prefer "not equal to": ${format(match)}`);
}
for (const match of text.matchAll(/\bthe \S+ argument\b/g)) {
  error(`Drop 'the' and 'argument': ${format(match)}`);
}
for (const match of text.matchAll(/\bnot greater\b/g)) {
  error(`Prefer "less or equal" to "not greater" (etc):  ${format(match)}`);
}

// [WebNN] Ensure MLOperandDescriptor's shape is linked, not MLOperand's.
// This looks for variables containing 'desc' (descriptor, desc2, etc).
// FUTURE: Implement a "type checker" for specs.
for (const match of source.matchAll(/(\|\w*desc\w*\|)'s \[=MLOperand\/shape=\]/ig)) {
  error(`Use ${match[1]}.{{MLOperandDescriptor/shape}} not MLOperand's shape: ${format(match)}`);
}

// [Generic] Look for missing dict-member dfns
// If there is a definition for a dictionary member, Bikeshed automagically
// links the IDL to that definition. Otherwise, it makes the IDL the definition
// itself - which we consider an error.
for (const element of root.querySelectorAll(
       IDL_BLOCK_SELECTOR + DESCENDANT_COMBINATOR + DICTMEMBER_DFN_SELECTOR)) {
  error(`Dictionary member missing dfn: ${element.innerText}`);
}

// [WebNN] Look for suspicious stuff in algorithm steps
for (const element of root.querySelectorAll(ALGORITHM_STEP_SELECTOR)) {
  // [] used for anything but indexing, slots, and refs
  // Exclude \w[ for indexing (e.g. shape[n])
  // Exclude [[ for inner slots (e.g. [[name]])
  // Exclude [A for references (e.g. [WEBIDL])
  for (const match of element.innerText.matchAll(/(?<!\w|\[|\]|«)\[(?!\[|[A-Z])/g)) {
    error(`Non-index use of [] in algorithm: ${format(match)}`);
  }
  // | in the DOM is likely an unclosed variable, since we don't use symbols for
  // absolute (|n|) , logical or (a || b) or bitwise or (a | b).
  for (const match of element.innerText.matchAll(/\|/g)) {
    error(`Unclosed variable in algorithm: ${format(match)}`);
  }
}

// [Generic] Ensure vars are method/algorithm arguments, or initialized correctly
// Every variable should be a method argument, an argument to an abstract method,
// or initialized with "Let".
for (const algorithm of root.querySelectorAll(ALGORITHM_SELECTOR)) {
  const vars = algorithm.querySelectorAll(VAR_SELECTOR);
  const seen = new Set();
  for (const v of vars) {
    const name = v.innerText.trim().replaceAll(/\s+/g, ' ');

    if (v.parentNode.tagName === 'CODE' && v.parentNode.parentNode.tagName === 'DFN') {
      // Argument definition for IDL method algorithms
      // e.g. "The compute(graph, inputs, outputs) method steps are:"
      seen.add(name);
    } else if (v.parentNode.querySelectorAll('dfn').length) {
      // Argument definition for abstract algorithms
      // e.g. "To execute graph, given MLGraph graph ..."
      seen.add(name);
    } else {
      const text = v.parentNode.innerText.trim().replaceAll(/\s+/g, ' ');
      const patterns = [
        // "Let var be ..."
        // "Let var1 be ... and and var2 be ..."
        'let( .* and)? ' + name + ' be',

        // "Let var given ... be ..." (for lambdas)
        'let ' + name + ' given .* be',

        // "Let « ..., var, ... » be ..."
        'let «( .*,)? ' + name + '(, .*)? » be',

        // "For each var ..."
        // "For each type var ..."
        // "For each key → var ..."
        'for each( \\w+| \\w+ →)? ' + name,
      ];
      if (patterns.some(p => new RegExp('\\b' + p + '\\b', 'i').test(text))) {
        // Variable declaration/initialization
        seen.add(name);
      } else if (new RegExp('\\bgiven .* \\b' + name + '\\b', 'i').test(text)) {
        // Lambda argument declarations
        // e.g. "Let validationSteps given MLOperandDescriptor descriptor be..."
        seen.add(name);
      } else if (!seen.has(name)) {
        error(`Uninitialized variable "${name}" in "${algorithm.getAttribute('data-algorithm')}": ${text}`);
        seen.add(name);
      }
    }
  }
}

// [Generic] Eschew vars outside of algorithms.
// This is important as Bikeshed has some smarts around variable usage, e.g.
// flagging a variable only referenced once within an algorithm. "Global"
// variables confuse this.
const algorithmVars = new Set(root.querySelectorAll(
  ALGORITHM_SELECTOR + DESCENDANT_COMBINATOR + VAR_SELECTOR));
for (const v of root.querySelectorAll(VAR_SELECTOR)
       .filter(v => !algorithmVars.has(v))) {
  error(`Variable outside of algorithm: ${v.innerText}`);
}

// [Generic] Algorithms should either throw or reject, never both.
for (const algorithm of root.querySelectorAll(ALGORITHM_SELECTOR)) {
  const name = algorithm.getAttribute('data-algorithm');
  const terms = {
    throw: 'exception',
    resolve: 'promise',
    resolved: 'promise',
    reject: 'promise',
    rejected: 'promise',
  };
  const other = {
    exception: 'promise',
    promise: 'exception',
  };
  // Look for any of the terms above as stand-alone words within step text.
  const re = RegExp('\\b(' + Object.keys(terms).join('|') + ')\\b', 'g');
  const seen = new Set();

  for (const match of algorithm.innerText.matchAll(re)) {
    // Map the term to the type ('exception' or 'promise').
    const type = terms[match[1]];

    // If we saw the other type in use in this algorithm, that's an error.
    if (seen.has(other[type])) {
      error(`Algorithm "${name}" mixes throwing with promises: ${format(match)}`);
      break;
    }
    seen.add(type);
  }
}

// [WebNN] Prevent accidental normative references to other specs. This reports
// an error if there is a normative reference to any spec *other* than these
// ones. This helps avoid an autolink like [=object=] adding an unexpected
// reference to [FILEAPI]. Add to this list if a new normative reference is
// intended.
const NORMATIVE_REFERENCES = new Set([
  '[ECMASCRIPT]',
  '[HTML]',
  '[INFRA]',
  '[NUMPY-BROADCASTING-RULE]',
  '[PERMISSIONS-POLICY-1]',
  '[RFC2119]',
  '[WEBGPU]',
  '[WEBIDL]',
]);

for (const term of root.querySelectorAll('#normative + dl > dt')) {
  const ref = term.innerText.trim();
  if (!NORMATIVE_REFERENCES.has(ref)) {
    errorHtml(`Unexpected normative reference to ${ref}`, term);
  }
}

// [Generic] Detect syntax errors in JS.
// This works by grabbing JS examples and parsing them (like `eval()`). Note
// that since examples usually lack full context (e.g. device creation), and the
// Node.js environment doesn't support WebNN anyway, this does not execute the
// JS, so only syntax errors are found.
for (const pre of root.querySelectorAll('pre.highlight:not(.idl)')) {
  const script = innerText(pre);
  try {
    const f = AsyncFunction([], '"use strict";' + script);
  } catch (ex) {
    error(`Invalid script: ${ex.message}: ${script.substr(0, 20)}`);
  }
}

// [Generic] Ensure algorithm steps end in '.' or ':'.
for (const p of root.querySelectorAll(ALGORITHM_STEP_SELECTOR)) {
  const match = p.innerText.match(/[^.:]$/);
  if (match) {
    error(`Algorithm steps should end with '.' or ':': ${format(match)}`);
  }
}

// [WebNN] Don't return undefined from methods - it is implied. Conditionally
// returning undefined may make sense in some algorithms, so this is considered
// a WebNN-specific rule.
for (const p of root.querySelectorAll(ALGORITHM_STEP_SELECTOR)) {
  if (p.innerText === 'Return undefined.') {
    error(`Unnecessary algorithm step: ${p.innerText}`);
  }
}

// [Generic] Ensure algorithm steps with "If" have a "then".
for (const p of root.querySelectorAll(ALGORITHM_STEP_SELECTOR)) {
  const text = p.innerText;
  const match = text.match(/\bIf\b/);
  const match2 = text.match(/, then\b/);
  if (match && !match2) {
    error(`Algorithm steps with 'If' should have ', then': ${format(match)}`);
  }
}

// [Generic] Ensure "Otherwise" is followed by expected punctuation.
for (const p of root.querySelectorAll(ALGORITHM_STEP_SELECTOR)) {
  const text = p.innerText;
  const match = text.match(/^Otherwise[^,:]/);
  if (match) {
    error(`In algorithm steps 'Otherwise' should be followed by ':' or ',': ${format(match)}`);
  }
}

// [Generic] Avoid links to [=list/empty=] when [=list/is empty=] is intended.
for (const match of source.matchAll(/is( not)? \[=(list\/|stack\/|queue\/|)empty=\]/g)) {
  error(`Link to 'is empty' (adjective) not 'empty' (verb): ${format(match)}`);
}

// [Generic] Ensure every method <dfn> is correctly associated with an
// interface. Bikeshed will let you define a method for an interface that isn't
// itself defined, so this helps catch places where an interface is renamed but
// not all references have been updated.
const interfaces = new Set(
  root.querySelectorAll(INTERFACE_DFN_SELECTOR).map(e => e.innerText));
for (const dfn of root.querySelectorAll(METHOD_DFN_SELECTOR)) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  if (!dfnFor || !interfaces.has(dfnFor)) {
    error(`Method definition '${dfn.innerText}' for undefined '${dfnFor}'`);
  }
}

// [Generic] Ensure every IDL argument is linked to a definition.
for (const dfn of root.querySelectorAll(
       IDL_BLOCK_SELECTOR + DESCENDANT_COMBINATOR + ARGUMENT_DFN_SELECTOR)) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  error(`Missing <dfn argument for="${dfnFor}">${dfn.innerText}</dfn> (or equivalent)`);
}

// [Generic] Ensure every argument <dfn> is correctly associated with a method.
// This tries to catch extraneous definitions, e.g. after an arg is removed.
for (const dfn of root.querySelectorAll(ARGUMENT_DFN_SELECTOR)) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  if (!dfnFor.split(/\b/).includes(dfn.innerText)) {
    error(`Argument definition '${dfn.innerText}' doesn't appear in '${dfnFor}'`);
  }
}

// [WebNN] Try to catch type mismatches like |tensor|.{{MLGraph/...}}. Note that
// the test is keyed on the variable name; variables listed here are not
// validated.
// FUTURE: Implement a "type checker" for specs.
for (const match of source.matchAll(/\|(\w+)\|\.{{(\w+)\/.*?}}/g)) {
  const [_, v, i] = match;
  [['MLTensor', ['tensor']],
   ['MLGraph', ['graph']],
   ['MLOperand', ['operand', 'input', 'output0', 'output1', 'output2']],
   ['MLOperandDescriptor', ['descriptor', 'desc', 'inputDescriptor']],
  ].forEach(pair => {
    const [iname, vnames] = pair;
    vnames.forEach(vname => {
      if (v === vname && i !== iname) {
        error(`Variable name '${v}' and type '${i}' do not match: ${
          format(match)}`);
      }
    });
  });
}

// [WebNN] Verify that linked constraints table IDs are reasonable. Bikeshed
// will flag any broken links; this just tries to ensure that links within the
// algorithm go to that algorithm's associated table.
for (const algorithm of root.querySelectorAll(
       '.algorithm[data-algorithm-for=MLGraphBuilder]')) {
  const name = algorithm.getAttribute('data-algorithm');
  // Restrict to methods, e.g. "foo(", and capture its name (in `RegExp.$1`)
  if (name.match(/^(\w+)\(/)) {
    const method = RegExp.$1;
    // Look for links to "#constraints-..."
    for (const href of algorithm.querySelectorAll('a')
           .map(a => a.getAttribute('href'))
           .filter(href => href.match(/#constraints-/))) {
      // Allow either exact case or lowercase match for table ID.
      if (
        href !== '#constraints-' + method &&
        href !== '#constraints-' + method.toLowerCase()) {
        error(`Steps for ${method}() link to ${href}`);
      }
    }
  }
}

// [WebNN] Ensure constraints tables use linking not styling
for (const table of root.querySelectorAll('table.data').filter(e => e.id.startsWith('constraints-'))) {
  // Look for `<em>identifier</em>` which is wrong, except for `output`.
  for (const match of table.innerHTML.matchAll(/<em>(?!output)(\w+)<\/em>/ig)) {
    error(`Constraints table should link not style args: ${format(match)}`);
  }
}

const dictionaryTypes =
  idl_ast.filter(o => o.type === 'dictionary').map(o => o.name);

// [Generic] Ensure JS objects are created with explicit realm
// Looks for " a new promise" not followed by " in realm".
for (const match of text.matchAll(/ a new promise\b(?! in realm)/g)) {
  error(`Promise creation must specify realm: ${format(match)}`);
}
// [Generic] Ensure JS objects are created with explicit realm
// Looks for " be a new XYZ" not followed by "in realm".
for (const match of text.matchAll(/ be a new ([A-Z]\w+)\b(?! in realm)/g)) {
  const type = match[1];
  // Dictionaries are just maps, so they don't need a realm.
  if (dictionaryTypes.includes(type))
    continue;
  error(`Object creation must specify realm: ${format(match)}`);
}

globalThis.process.exit(exitCode);
