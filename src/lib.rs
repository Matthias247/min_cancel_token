/// Error type which is returned if a cancellation token is cancelled
#[derive(Debug)]
pub struct CancellationError;

pub struct CancellationGuardVtable {
    drop: unsafe fn(data: *const (), func: &mut CancellationFunc),
}

/// Guard for a registered cancellation function
///
/// If the guard is dropped, the cancellation handler will be removed
pub struct CancellationGuard<'a> {
    func: &'a mut CancellationFunc<'a>,
    data: *const (),
    vtable: &'static CancellationGuardVtable,
}

impl<'a> Drop for CancellationGuard<'a> {
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(self.data, self.func) };
    }
}

/// A function that will be executed on cancellation
///
/// This is a tiny wrapper around a `FnMut` closure which allows to intrusively
/// link multiple closures.
pub struct CancellationFunc<'a> {
    inner: Option<&'a mut (dyn FnMut() + Sync)>,
    prev: *const (),
    next: *const (),
}

impl<'a> CancellationFunc<'a> {
    unsafe fn from_raw(raw: *const ()) -> &'a mut Self {
        let a = raw as *const Self;
        std::mem::transmute(a)
    }

    fn into_raw(&mut self) -> *const () {
        self as *const Self as _
    }

    pub fn new(func: &'a mut (dyn FnMut() + Sync)) -> Self {
        Self {
            inner: Some(func),
            prev: std::ptr::null(),
            next: std::ptr::null(),
        }
    }
}

/// A `CancellationToken` provides information whether a flow of execution is expected
/// to be cancelled.
///
/// There are 2 ways to interact with `CancellationToken`:
/// 1. The token can be queried on whether the flow is cancelled
/// 2. A callback can be registered if thhe flow is cancelled.
pub trait CancellationToken {
    /// Performs a one-time check whether the flow of execution is cancelled.
    ///
    /// Returns an error if cancellation is initiated
    fn error_if_cancelled(&self) -> Result<(), CancellationError>;

    /// Registers a cancellation handler, which will be invoked when the execution flow
    /// is cancelled.
    /// The cancellation handler can be called from any thread which initiates the cancellation.
    /// If the flow is already cancelled when this function is called, the cancellation
    /// handler will be called synchronously.
    /// The function returns a guard which can be used to unregister the cancellation handler.
    /// After the guard is dropped, the handler is guaranteed not be called anymore.
    fn on_cancellation<'a>(&self, func: &'a mut CancellationFunc<'a>) -> CancellationGuard<'a>;
}

thread_local! {
    pub static CURRENT_CANCELLATION_TOKEN: std::cell::RefCell<Option<&'static dyn CancellationToken>> = std::cell::RefCell::new(None);
}

/// Executes a function that gets access to the current cancellation token
pub fn with_current_cancellation_token<R>(func: impl FnOnce(&dyn CancellationToken) -> R) -> R {
    CURRENT_CANCELLATION_TOKEN.with(|token| {
        let x = &*token.borrow();
        match x {
            Some(token) => func(*token),
            None => func(&UncancellableToken::default()),
        }
    })
}

/// Replaces the currently active (thread-local) cancellation token with the provided one,
/// and executes the given function.
/// Once the scope ends, the current cancellation token will be reset to the previous one.
pub fn with_cancellation_token<'a, R>(
    token: &'a dyn CancellationToken,
    func: impl FnOnce() -> R,
) -> R {
    // Note that `'static` is a hack here to avoid having to specify the outer (unknown) lifetimes.
    // Since we are guaranteed to reset the token before the lifetime ends and
    // don't copy it anywhere else, this is ok.
    struct RevertToOldTokenGuard<'a> {
        prev: Option<&'static dyn CancellationToken>,
        storage: &'a std::cell::RefCell<Option<&'static dyn CancellationToken>>,
    }

    impl<'a> Drop for RevertToOldTokenGuard<'a> {
        fn drop(&mut self) {
            let mut guard = self.storage.borrow_mut();
            *guard = self.prev;
        }
    }

    CURRENT_CANCELLATION_TOKEN.with(|storage| {
        let mut guard = storage.borrow_mut();
        let static_token: &'static dyn CancellationToken = unsafe { std::mem::transmute(token) };

        let prev = std::mem::replace(&mut *guard, Some(static_token));
        drop(guard);

        // Revert to the last token once the function ends
        // This guard makes sure we are adhering to to the non static lifetime
        let _revert_guard = RevertToOldTokenGuard { prev, storage };

        func()
    })
}

/// An implementation of `CancellationToken` which will never signal the
/// cancelled state
#[derive(Debug, Default)]
pub struct UncancellableToken {}

fn noop_drop(_data: *const (), _func: &mut CancellationFunc) {}

fn noop_vtable() -> &'static CancellationGuardVtable {
    &CancellationGuardVtable { drop: noop_drop }
}

impl CancellationToken for UncancellableToken {
    fn error_if_cancelled(&self) -> Result<(), CancellationError> {
        Ok(())
    }

    fn on_cancellation<'a>(&self, func: &'a mut CancellationFunc<'a>) -> CancellationGuard<'a> {
        CancellationGuard {
            func,
            data: std::ptr::null(),
            vtable: noop_vtable(),
        }
    }
}

pub mod std_impl {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Inner state of the `std` `CancellationToken` implementation
    struct State {
        /// Whether cancellation was initiated
        pub cancelled: bool,
        /// Linked list of cancellation callbacks
        pub first_func: *const (),
        /// Linked list of cancellation callbacks
        pub last_func: *const (),
    }

    fn std_cancellation_token_drop(data: *const (), func: &mut CancellationFunc) {
        let state: Arc<Mutex<State>> = unsafe { Arc::from_raw(data as _) };

        let mut guard = state.lock().unwrap();
        if guard.cancelled {
            // The token was already cancelled and the callback was called
            // This also means the function should already have been removed from
            // the linked list
            assert!(func.prev.is_null());
            assert!(func.next.is_null());
            return;
        }

        unsafe {
            if func.prev.is_null() {
                // This must be the first function that is registered
                guard.first_func = func.next;
                if !guard.first_func.is_null() {
                    let mut first = CancellationFunc::from_raw(guard.first_func);
                    first.prev = std::ptr::null();
                } else {
                    // The list is drained
                    guard.last_func = std::ptr::null();
                }
                func.next = std::ptr::null();
            } else {
                // There exists a previous function, since its not the first
                let mut prev = CancellationFunc::from_raw(func.prev);
                prev.next = func.next;
                if !func.next.is_null() {
                    let mut next = CancellationFunc::from_raw(func.next);
                    next.prev = func.prev;
                }

                func.next = std::ptr::null();
                func.prev = std::ptr::null();
            }
        }

        std::mem::drop(data);
    }

    fn std_cancellation_token_vtable() -> &'static CancellationGuardVtable {
        &CancellationGuardVtable {
            drop: std_cancellation_token_drop,
        }
    }

    pub struct StdCancellationToken {
        state: Arc<Mutex<State>>,
    }

    impl StdCancellationToken {}

    impl CancellationToken for StdCancellationToken {
        fn error_if_cancelled(&self) -> Result<(), crate::CancellationError> {
            if self.state.lock().unwrap().cancelled {
                Err(CancellationError)
            } else {
                Ok(())
            }
        }

        fn on_cancellation<'a>(&self, func: &'a mut CancellationFunc<'a>) -> CancellationGuard<'a> {
            let mut guard = self.state.lock().unwrap();
            if guard.cancelled {
                if let Some(func) = (&mut func.inner).take() {
                    (func)();
                }
                return CancellationGuard {
                    data: std::ptr::null(),
                    vtable: noop_vtable(),
                    func,
                };
            }

            func.next = std::ptr::null();
            func.prev = std::ptr::null();
            if guard.first_func.is_null() {
                // Only function in the list
                guard.first_func = func.into_raw();
                guard.last_func = func.into_raw();
            } else {
                unsafe {
                    // This must exist, since its not the only function in the list
                    let mut last = CancellationFunc::from_raw(guard.last_func);
                    last.next = func.into_raw();
                    func.prev = last.into_raw();
                    guard.last_func = func.into_raw();
                }
            }

            CancellationGuard {
                data: Arc::into_raw(self.state.clone()) as _,
                vtable: std_cancellation_token_vtable(),
                func,
            }
        }
    }

    pub struct StdCancellationTokenSource {
        state: Arc<Mutex<State>>,
    }

    unsafe impl Send for StdCancellationTokenSource {}
    unsafe impl Sync for StdCancellationTokenSource {}

    impl StdCancellationTokenSource {
        pub fn new() -> StdCancellationTokenSource {
            StdCancellationTokenSource {
                state: Arc::new(Mutex::new(State {
                    cancelled: false,
                    first_func: std::ptr::null(),
                    last_func: std::ptr::null(),
                })),
            }
        }

        pub fn token(&self) -> StdCancellationToken {
            StdCancellationToken {
                state: self.state.clone(),
            }
        }

        pub fn cancel(&self) {
            let mut guard = self.state.lock().unwrap();
            if guard.cancelled {
                return;
            }
            guard.cancelled = true;

            while !guard.first_func.is_null() {
                unsafe {
                    let mut first = CancellationFunc::from_raw(guard.first_func);
                    guard.first_func = first.next;
                    first.prev = std::ptr::null();
                    first.next = std::ptr::null();
                    if let Some(func) = first.inner.take() {
                        (func)();
                    }
                }
            }
            guard.last_func = std::ptr::null();
        }
    }
}

/// Utilities for working with cancellation tokens
pub mod utils {
    use super::*;

    pub fn wait_cancelled(token: &dyn CancellationToken) {
        let mtx = std::sync::Mutex::new(false);
        let cv = std::sync::Condvar::new();

        let func = &mut || {
            let mut guard = mtx.lock().unwrap();
            *guard = true;
            drop(guard);
            cv.notify_all();
        };
        let mut wait_func = CancellationFunc::new(func);
        let _guard = token.on_cancellation(&mut wait_func);

        let mut cancelled = mtx.lock().unwrap();
        while !*cancelled {
            cancelled = cv.wait(cancelled).unwrap();
        }
    }

    pub fn wait_cancelled_polled(token: &dyn CancellationToken) {
        let is_cancelled = std::sync::atomic::AtomicBool::new(false);

        let func = &mut || {
            is_cancelled.store(true, std::sync::atomic::Ordering::Release);
        };
        let mut wait_func = CancellationFunc::new(func);
        let _guard = token.on_cancellation(&mut wait_func);

        while !is_cancelled.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub async fn await_cancelled(token: &dyn CancellationToken) {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::Mutex;
        use std::task::{Context, Poll, Waker};

        struct CancelFut<'a> {
            token: &'a dyn CancellationToken,
            waker: &'a Mutex<Option<Waker>>,
        }

        impl<'a> Future for CancelFut<'a> {
            type Output = ();

            fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<<Self as Future>::Output> {
                match self.token.error_if_cancelled() {
                    Ok(()) => {
                        let mut guard = self.waker.lock().unwrap();
                        *guard = Some(cx.waker().clone());

                        // TODO: Theres a race here, and the waker might just have been
                        // installed after the token was cancelled

                        Poll::Pending
                    }
                    Err(_) => Poll::Ready(()),
                }
            }
        }

        // TODO: A Mutex requires a heap allocation, and is probably not required
        // here. Something like `AtomicWaker` should work.
        let waker_store = Mutex::<Option<Waker>>::new(None);

        let mut on_cancel = || {
            let mut guard = waker_store.lock().unwrap();
            if let Some(waker) = guard.take() {
                waker.wake();
            }
        };
        let mut wait_func = CancellationFunc::new(&mut on_cancel);
        let _guard = token.on_cancellation(&mut wait_func);

        let fut = CancelFut {
            token,
            waker: &waker_store,
        };

        fut.await
    }
}

#[cfg(test)]
mod tests {
    use super::std_impl::*;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn simple_cancel() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();

        assert!(token.error_if_cancelled().is_ok());
        source.cancel();
        assert!(token.error_if_cancelled().is_err());
    }

    #[test]
    fn test_token() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();

        let dyn_token: &dyn CancellationToken = &token;

        let (sender, receiver) = std::sync::mpsc::sync_channel(1);

        let start = Instant::now();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            source.cancel();
        });

        let mut func = || {
            sender.send(true).unwrap();
        };
        let mut cancel_func = CancellationFunc::new(&mut func);

        let _guard = dyn_token.on_cancellation(&mut cancel_func);

        let _ = receiver.recv();

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(1));
    }

    #[test]
    fn test_wait_cancelled_immediately() {
        let source = StdCancellationTokenSource::new();
        source.cancel();
        let token = source.token();

        let dyn_token: &dyn CancellationToken = &token;

        let start = Instant::now();

        utils::wait_cancelled(dyn_token);

        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(50));
    }

    #[test]
    fn test_wait_cancelled() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();

        let dyn_token: &dyn CancellationToken = &token;

        let start = Instant::now();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            source.cancel();
        });

        utils::wait_cancelled(dyn_token);

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(1));
    }

    #[test]
    fn test_wait_cancelled_polled() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();

        let dyn_token: &dyn CancellationToken = &token;

        let start = Instant::now();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            source.cancel();
        });

        utils::wait_cancelled_polled(dyn_token);

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(1));
    }

    #[test]
    fn test_await_cancelled_immediately() {
        futures::executor::block_on(async {
            let source = StdCancellationTokenSource::new();
            source.cancel();
            let token = source.token();
            let dyn_token: &dyn CancellationToken = &token;

            let start = Instant::now();

            utils::await_cancelled(dyn_token).await;

            let elapsed = start.elapsed();
            assert!(elapsed < Duration::from_millis(50));
        });
    }

    #[test]
    fn test_await_cancelled() {
        futures::executor::block_on(async {
            let source = StdCancellationTokenSource::new();
            let token = source.token();

            let dyn_token: &dyn CancellationToken = &token;

            let start = Instant::now();

            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(1));
                source.cancel();
            });

            utils::await_cancelled(dyn_token).await;

            let elapsed = start.elapsed();
            assert!(elapsed >= Duration::from_secs(1));
        });
    }

    #[test]
    fn unregister_before_cancel() {
        for token1_to_drop in 0..4 {
            for token2_to_drop in 0..4 {
                let source = StdCancellationTokenSource::new();
                let tokens = (0..4).map(|_| source.token()).collect::<Vec<_>>();

                let counter = Arc::new(AtomicUsize::new(0));

                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(1));
                    source.cancel();
                });

                let mut func_1 = || {
                    counter.fetch_add(1, Ordering::SeqCst);
                };
                let mut func_2 = || {
                    counter.fetch_add(1, Ordering::SeqCst);
                };
                let mut func_3 = || {
                    counter.fetch_add(1, Ordering::SeqCst);
                };
                let mut func_4 = || {
                    counter.fetch_add(1, Ordering::SeqCst);
                };
                let mut cancel_func_1 = CancellationFunc::new(&mut func_1);
                let mut cancel_func_2 = CancellationFunc::new(&mut func_2);
                let mut cancel_func_3 = CancellationFunc::new(&mut func_3);
                let mut cancel_func_4 = CancellationFunc::new(&mut func_4);

                let mut guards = vec![None, None, None, None];
                guards[0] = Some(tokens[0].on_cancellation(&mut cancel_func_1));
                guards[1] = Some(tokens[1].on_cancellation(&mut cancel_func_2));
                guards[2] = Some(tokens[2].on_cancellation(&mut cancel_func_3));
                guards[3] = Some(tokens[3].on_cancellation(&mut cancel_func_4));

                guards[token1_to_drop] = None;
                guards[token2_to_drop] = None;

                std::thread::sleep(Duration::from_secs(2));
                let expected = if token1_to_drop == token2_to_drop {
                    3
                } else {
                    2
                };
                assert_eq!(counter.load(Ordering::SeqCst), expected);
            }
        }
    }

    #[test]
    fn test_thread_local_cancellation() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();
        let dyn_token: &dyn CancellationToken = &token;

        let start = Instant::now();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            source.cancel();
        });

        with_cancellation_token(dyn_token, || {
            with_current_cancellation_token(|token| {
                utils::wait_cancelled(token);
            })
        });

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(1));
    }

    #[test]
    fn test_nested_cancellation() {
        let source = StdCancellationTokenSource::new();
        let token = source.token();
        let dyn_token: &dyn CancellationToken = &token;

        let start = Instant::now();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            source.cancel();
        });

        with_cancellation_token(dyn_token, || {
            let next_source = StdCancellationTokenSource::new();
            let next_token = next_source.token();

            let mut cancel_func = || {
                next_source.cancel();
            };
            let mut cancel_func = CancellationFunc::new(&mut cancel_func);
            let _guard =
                with_current_cancellation_token(|token| token.on_cancellation(&mut cancel_func));

            with_cancellation_token(&next_token, || {
                let third_source = StdCancellationTokenSource::new();
                let third_token = third_source.token();

                let mut cancel_func = || {
                    third_source.cancel();
                };
                let mut cancel_func = CancellationFunc::new(&mut cancel_func);
                let _guard = with_current_cancellation_token(|token| {
                    token.on_cancellation(&mut cancel_func)
                });

                with_cancellation_token(&third_token, || {
                    with_current_cancellation_token(|token| {
                        futures::executor::block_on(async {
                            utils::await_cancelled(token).await;
                        });
                    });
                });
            });
        });

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(1));
    }
}
