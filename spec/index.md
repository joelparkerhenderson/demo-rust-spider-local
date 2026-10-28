# Demo UK NHS LLM

Demonstration of how to create a United Kingdom (UK) National Health Service
(NHS) Large Language Model (LLM) by using a small set of public web pages,
tokenizing them, and then creating a result.

- This is a simple example suitable for demonstrating concepts from the ground up.

- This is a personal projec, not an official project.

- This project is not medically tested, not for medical use.

Steps:

1. Read public web pages at https://www.nhs.uk/conditions/

2. Parse each page from HTML into the main text.

## Coding conventions

Prefer pure functions (not `self` methods).

The concept: a trait impl (and an `impl` block) is for state-related methods.
Prefer pure functions in a separate file, named `*_functions.rs`.

| Where | What belongs there |
|-------|--------------------|
| `tensor.rs`, `tokenizer.rs`, ... (the struct's file) | The struct, its `impl` block and its trait impls: methods that are about the struct's own state. They build it (`new`, `default`, `from_merges`), read it (`vocab_size`, `stats`), or work directly on its fields (`reshape`). Trait impls (`Default`, `Display`, `Clone`, ...) say what the type is. |
| `tensor_functions.rs`, `tokenizer_functions.rs`, ... (a separate file) | Pure functions: the operations and algorithms. They take the struct (or just the data they need) as an argument. |

So a struct's file stays small and answers "what is this thing, and what state
does it hold?", and the `*_functions.rs` file answers "what can be computed
with it?". When you are about to add a method, ask whether it is about the
struct's state. If not, write a pure function in the separate file instead.

A pure function takes everything it needs as arguments and returns its result.
It reads no hidden state and changes nothing except what it is explicitly given
(for example, a `&mut` argument). Put the real logic in such functions, and
pass only what the function needs, not a whole struct.

Prefer this:

```rust
fn matmul_parallel_blocked(a: &Tensor, b: &Tensor, m: usize, n: usize, k: usize) -> Tensor
fn normalize_axis(shape: &[usize], axis: isize, op: &str) -> usize
```

Avoid this:

```rust
impl Tensor {
    fn matmul_parallel_blocked(&self, other: &Tensor, m: usize, n: usize, k: usize) -> Tensor
    fn normalize_axis(&self, axis: isize, op: &str) -> usize
}
```

Why:

- Each function can be read on its own: what goes in, what comes out.
- Each function can be tested without building a whole struct first.
- Each function can be reused by other code.

If a method only calls a pure function, and does nothing else with the `self`
innards, remove the method in favor of the pure function. Callers call the
function directly, and the function takes the struct as an argument:

```rust
// Do this:
let c = matmul(&a, &b);        // pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor
let ids = encode(&tokenizer, text);

// Not this:
let c = a.matmul(&b);          // pub fn matmul(&self, other: &Tensor) -> Tensor
let ids = tokenizer.encode(text);
```

Put such functions in the `*_functions.rs` file next to the struct's file
(`tensor_functions.rs`, `tokenizer_functions.rs`). A struct whose fields these
functions need must make them `pub`.

Where it is reasonable to keep a method:

- Constructors and trait implementations (`new`, `default`, `Display`, ...).
- A method that does real work of its own with the struct's fields, beyond
  calling a pure function (for example `reshape`, `vocab_size`).
- A method that also does something with side effects, such as file I/O or
  printing (for example `save`, `load`, `analyze_vocabulary`), where the pure
  part is already a function it calls.

A function may take `&mut` of a struct to change it, such as
`train(&mut tokenizer, text, size)`; that is not a reason to keep a method.

Functions are public by default (`pub`). Make a function private only when
there is a specific reason to hide it.

## Test conventions

Prefer explicit variables "actual" and "expect", such as:

```rust
let actual = foo(x);
let expect = 1;
assert_eq!(actual, expect);
```

Prefer explicit intermediate variables, such as:

```rust
let url_str = "http://example.com/alfa/bravo/"
let html_str = "<html><body><main>hello world</main></body></html>";
let foo = foo(url_str, html_str);
```
