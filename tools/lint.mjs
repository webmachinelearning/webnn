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

'use strict';
import fs from 'node:fs/promises';
import {parse} from 'node-html-parser';

// --------------------------------------------------
// Process options
// --------------------------------------------------

const options = {
  verbose: false,
  bikeshed: 'index.bs',
  html: 'index.html',
};

// First two args are interpreter and script
globalThis.process.argv.slice(2).forEach((arg, index, array) => {
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

log(`loading Bikeshed source "${options.bikeshed}"...`);
const source = await fs.readFile(options.bikeshed, 'utf8');

log(`loading generated HTML "${options.html}"...`);
let file = await fs.readFile(options.html, 'utf8');

log('massaging HTML...');
// node-html-parser doesn't understand that some elements are mutually self-closing;
// tweak the source using regex magic.
[{tags: ['dt', 'dd'], containers: ['dl']},
 {tags: ['thead', 'tbody', 'tfoot'], containers: ['table']},
 {tags: ['tr'], containers: ['thead', 'tbody', 'tfoot', 'table']},
].forEach(({tags, containers}) => {
  const re = new RegExp(
    '(<(' + tags.join('|') + ')\\b[^>]*>)' +
      '(.*?)' +
      '(?=<(' + tags.join('|') + '|/(' + containers.join('|') + '))\\b)',
    'sg');
  file = file.replaceAll(
    re, (_, opener, tag, content) => `${opener}${content}</${tag}>`);
});

log('parsing HTML...');
const root = parse(file, {
  blockTextElements: {
    // Explicitly don't list <pre> to force children to be parsed.
    // See https://github.com/taoqf/node-html-parser/issues/78

    // Explicitly list <script> and <style> otherwise remove() leaves
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

let exitCode = 0;
function error(message) {
  console.error(message);
  exitCode = 1;
}


function format(match) {
  const CONTEXT = 20;

  let prefix = match.input.substring(match.index - CONTEXT, match.index)
                     .split(/\n/)
                     .pop();
  let suffix = match.input.substr(match.index + match[0].length, CONTEXT)
                     .split(/\n/)
                     .shift();
  let infix = match[0];

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

const AsyncFunction = async function() {}.constructor;

// --------------------------------------------------
// Checks
// --------------------------------------------------

log('running checks...');

const ALGORITHM_STEP_SELECTOR = '.algorithm li > p:not(.issue)';

// Checks can operate on:
// * `source` - raw Bikeshed markdown source
// * `html` - HTML source, with style/script removed
// * `text` - rendered text content
// * `root.querySelectorAll()` - operate on DOM-like nodes

// Checks are marked with one of these tags:
// * [Generic] - could apply to any spec
// * [WebNN] - very specific to the WebNN spec

// [Generic] Look for merge markers
for (const match of source.matchAll(/<{7}|>{7}|^={7}$/mg)) {
  error(`Merge conflict marker: ${format(match)}`);
}

// [Generic] Look for residue of unterminated auto-links in rendered text
for (const match of text.matchAll(/({{|}}|\[=|=\])/g)) {
  error(`Unterminated autolink: ${format(match)}`);
}

// [Generic] Look for duplicate words (in source, since [=realm=] |realm| is okay)
for (const match of html.matchAll(/(?:^|\s)(\w+) \1(?:$|\s)/ig)) {
  error(`Duplicate word: ${format(match)}`);
}

// [Generic] Verify IDL lines wrap to avoid horizontal scrollbars
const MAX_IDL_WIDTH = 88;
for (const idl of root.querySelectorAll('pre.idl')) {
  idl.innerText.split(/\n/).forEach(line => {
    line = line.replace(/&lt;/g, '<'); // parser's notion of "innerText" is weird
    if (line.length > MAX_IDL_WIDTH) {
      error(`Overlong IDL: ${line}`);
    }
  });
}

// [WebNN] Look for undesired punctuation
for (const match of text.matchAll(/(::|×|÷|∗|−)/g)) {
  error(`Bad punctuation: ${format(match)}`);
}

// [WebNN] Look for undesired entity usage
for (const match of source.matchAll(/&(\w+);/g)) {
  if (!['amp', 'lt', 'gt', 'quot'].includes(match[1])) {
    error(`Avoid entities: ${format(match)}`);
  }
}

// [WebNN] Look for undesired phrasing
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

// [WebNN] Look for incorrect use of shape for an MLOperandDescriptor
for (const match of source.matchAll(/(\|\w*desc\w*\|)'s \[=MLOperand\/shape=\]/ig)) {
  error(`Use ${match[1]}.{{MLOperandDescriptor/dimensions}} not shape: ${format(match)}`);
}

// [Generic] Look for missing dict-member dfns
for (const element of root.querySelectorAll('.idl dfn[data-dfn-type=dict-member]')) {
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
  // | is likely an unclosed variable
  for (const match of element.innerText.matchAll(/\|/g)) {
    error(`Unclosed variable in algorithm: ${format(match)}`);
  }
}

// [Generic] Ensure vars are method/algorithm arguments, or initialized correctly
for (const algorithm of root.querySelectorAll('.algorithm')) {
  const vars = algorithm.querySelectorAll('var');
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
const algorithmVars = new Set(root.querySelectorAll('.algorithm var'));
for (const v of root.querySelectorAll('var').filter(v => !algorithmVars.has(v))) {
  error(`Variable outside of algorithm: ${v.innerText}`);
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
    error(`Unexpected normative reference to ${ref}`);
  }
}

// [Generic] Detect syntax errors in JS.
for (const pre of root.querySelectorAll('pre.highlight:not(.idl)')) {
  const script = pre.innerText.replaceAll(/&amp;/g, '&')
                     .replaceAll(/&lt;/g, '<')
                     .replaceAll(/&gt;/g, '>');
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

// [Generic] Avoid incorrect links to list/empty.
for (const match of source.matchAll(/is( not)? \[=(list\/|stack\/|queue\/|)empty=\]/g)) {
  error(`Link to 'is empty' (adjective) not 'empty' (verb): ${format(match)}`);
}

// [Generic] Ensure every method dfn is correctly associated with an interface.
const interfaces = new Set(
  root.querySelectorAll('dfn[data-dfn-type=interface]').map(e => e.innerText));
for (const dfn of root.querySelectorAll('dfn[data-dfn-type=method]')) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  if (!dfnFor || !interfaces.has(dfnFor)) {
    error(`Method definition '${dfn.innerText}' for undefined '${dfnFor}'`);
  }
}

// [Generic] Ensure every IDL argument is linked to a definition.
for (const dfn of root.querySelectorAll('pre.idl dfn[data-dfn-type=argument]')) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  error(`Missing <dfn argument for="${dfnFor}">${dfn.innerText}</dfn> (or equivalent)`);
}

// [Generic] Ensure every argument dfn is correctly associated with a method.
// This tries to catch extraneous definitions, e.g. after an arg is removed.
for (const dfn of root.querySelectorAll('dfn[data-dfn-type=argument]')) {
  const dfnFor = dfn.getAttribute('data-dfn-for');
  if (!dfnFor.split(/\b/).includes(dfn.innerText)) {
    error(`Argument definition '${dfn.innerText}' doesn't appear in '${dfnFor}'`);
  }
}

// [WebNN] Try to catch type mismatches like |tensor|.{{MLGraph/...}}. Note that
// the test is keyed on the variable name; variables listed here are not
// validated.
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
  if (name.match(/^(\w+)\(/)) {
    const method = RegExp.$1;
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
  for (const match of table.innerHTML.matchAll(/<em>(?!output)(\w+)<\/em>/ig)) {
    error(`Constraints table should link not style args: ${format(match)}`);
  }
}

// TODO: Generate this from the IDL itself.
const dictionaryTypes = ['MLOperandDescriptor', 'MLContextLostInfo'];

// [Generic] Ensure JS objects are created with explicit realm
for (const match of text.matchAll(/ a new promise\b(?! in realm)/g)) {
  error(`Promise creation must specify realm: ${format(match)}`);
}
// [Generic] Ensure JS objects are created with explicit realm
for (const match of text.matchAll(/ be a new ([A-Z]\w+)\b(?! in realm)/g)) {
  const type = match[1];
  // Dictionaries are just maps, so they don't need a realm.
  if (dictionaryTypes.includes(type))
    continue;
  error(`Object creation must specify realm: ${format(match)}`);
}

globalThis.process.exit(exitCode);
