min-cancel-token
================

This crate provides a standarized interface/trait for a `CancellationToken`, which
can be used by libraries and applications to react to cancellation events.

The `CancellationToken` interface can be interacted with in 2 ways:
- The token can be synchronously queried on whether cancellation is initiated
- A callback can be registered, which will be called exactly when cancellation is
  initiated.

The overall interface thereby follows the design of C++ [https://en.cppreference.com/w/cpp/thread/stop_token](std::stop_token).
However only the interface is standardized here, and no concrete implementation.

This has the benefit that different platforms and applications can provide implementations
which fit them best. E.g.
- normal applications can use heap allocated `CancellationToken` implementations,
  which are here provided as part of the `std_impls` module.
- embedded applications for realtime sytems could use one statically allocated
  `CancellationToken` per task.
- async runtimes could embed `CancellationToken`s inside the task state for each task,
  and thereby enable graceful per-task cancellation.

## Use-cases:
- Handle application shutdown gracefully, e.g. when close buttons are pressed or signals are received.
- Stop long running computations if the user aborts them
- Stop subtasks if there is no longer a need for them. E.g. within a webserver, there
  is no need to perform actions anymore if the client which issused a request disconnected.

## Additional background

See [A case for CancellationTokens](https://gist.github.com/Matthias247/354941ebcc4d2270d07ff0c6bf066c64)

## Composability

CancellationTokens are designed to be composable. The cancellation of a higher level task (e.g. the whole application) can trigger the cancellation of lower level tasks (e.g. performing a database query). That again could lead to cancellation of a particular IO call, like waiting to read data from a TCP connection.

As a consequence of this, most parts of applications might not have to be cancellation aware - it will be sufficient if long-running lower-level calls (e.g. socket and disk IO) understand cancellation and return errors if cancellation is initiated.

## State

This repository and crate is currently an experiment and a base for discussion.
It's not in a final state, and the interface provided by the crate might change.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.