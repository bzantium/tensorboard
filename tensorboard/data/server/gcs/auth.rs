/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//! OAuth integration for GCS. Authentication is done via the `gcp_auth` library.
//!
//! `gcp_auth`'s implementation of authentication when specifying GOOGLE_APPLICATION_CREDENTIALS
//! only supports service account based authentication. If you are attempting to provide OAuth
//! refresh tokens or other gcloud credential based authentication mechanisms and running into issues
//! (e.g. unable to initialize authentication manager: CustomServiceAccountCredentials(Error("missing field private_key"))),
//! you should unset the GOOGLE_APPLICATION_CREDENTIALS environment variable, as the final authentication
//! mechanism in `gcp_auth` will look for credentials in the standard file location
//! (i.e. ~/.config/gcloud/application_default_credentials.json) and properly handle authentication.
//!
//! Useful resources:
//!
//!   - TensorFlow OAuth implementation: [`oauth_client.cc`], [`google_auth_provider.cc`]
//!   - [RFC 6749]: The OAuth 2.0 Authorization Framework
//!   - ["Refreshing Access Tokens"] OAuth guide
//!
//! [`oauth_client.cc`]: https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/core/platform/cloud/oauth_client.cc
//! [`google_auth_provider.cc`]: https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/core/platform/cloud/google_auth_provider.cc
//! [RFC 6749]: https://tools.ietf.org/html/rfc6749
//! ["Refreshing Access Tokens"]: https://www.oauth.com/oauth2-servers/access-tokens/refreshing-access-tokens/

use log::{debug, info, warn};
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;

use gcp_auth::AuthenticationManager;
use reqwest::blocking::RequestBuilder;

const SCOPES: &[&str] = &["https://www.googleapis.com/auth/cloud-platform"];

/// After this many consecutive 401 responses, recreate the AuthenticationManager
/// to pick up fresh credentials from disk.
const MAX_CONSECUTIVE_401S_BEFORE_RESET: u32 = 3;

/// Resettable wrapper around `AuthenticationManager` that allows re-initialization
/// when credentials expire or are rotated on disk.
struct ResettableAuthManager {
    inner: RwLock<Option<AuthenticationManager>>,
    generation: AtomicU64,
}

static AUTH_MANAGER: ResettableAuthManager = ResettableAuthManager {
    inner: RwLock::new(None),
    generation: AtomicU64::new(0),
};

/// Get a GCP Access Token using the `gcp_auth::AuthenticationManager`.
///
/// Returns the token along with the current manager generation, which can be used
/// to avoid redundant resets when multiple threads detect stale credentials.
fn get_token() -> Result<(AccessToken, u64), gcp_auth::Error> {
    // Try to use existing manager under read lock.
    {
        let guard = AUTH_MANAGER.inner.read().expect("auth manager lock poisoned");
        if let Some(ref manager) = *guard {
            let generation = AUTH_MANAGER.generation.load(Ordering::Acquire);
            let token = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(manager.get_token(SCOPES))?;
            return Ok((AccessToken::new(Some(token)), generation));
        }
    }

    // Manager not yet initialized (or was reset); take write lock.
    let mut guard = AUTH_MANAGER.inner.write().expect("auth manager lock poisoned");
    // Double-check: another thread may have initialized while we waited.
    if guard.is_none() {
        let is_first_init = AUTH_MANAGER.generation.load(Ordering::Acquire) == 0;
        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(AuthenticationManager::new());
        match result {
            Ok(manager) => {
                *guard = Some(manager);
                AUTH_MANAGER.generation.fetch_add(1, Ordering::Release);
                if !is_first_init {
                    info!("Successfully re-initialized GCS authentication manager");
                }
            }
            Err(e) => {
                if is_first_init {
                    panic!("unable to initialize authentication manager: {}", e);
                }
                warn!(
                    "Failed to re-initialize authentication manager: {}; will retry next cycle",
                    e
                );
                return Err(e);
            }
        }
    }
    let generation = AUTH_MANAGER.generation.load(Ordering::Acquire);
    let token = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(guard.as_ref().unwrap().get_token(SCOPES))?;
    Ok((AccessToken::new(Some(token)), generation))
}

/// Resets the `AuthenticationManager` so the next `get_token()` call re-reads
/// credentials from disk. Only resets if `stale_generation` matches the current
/// generation, preventing redundant resets from concurrent threads.
fn reset_manager(stale_generation: u64) {
    let mut guard = AUTH_MANAGER.inner.write().expect("auth manager lock poisoned");
    let current = AUTH_MANAGER.generation.load(Ordering::Acquire);
    if current != stale_generation {
        debug!(
            "Auth manager already reset by another thread (generation {} -> {})",
            stale_generation, current
        );
        return;
    }
    warn!("Resetting GCS authentication manager to pick up fresh credentials");
    *guard = None;
    AUTH_MANAGER.generation.fetch_add(1, Ordering::Release);
}

/// A potentially active token. Use [`authenticate`][Self::authenticate] to add an `Authorization`
/// header to an outgoing request, fetching a fresh access token if necessary.
///
/// If it is `None`, then it has not been fetched yet.
///
/// A `TokenStore` may be freely shared among threads; it synchronizes internally if needed.
pub struct TokenStore {
    token: RwLock<Option<AccessToken>>,
    /// The manager generation when the current cached token was obtained.
    token_generation: RwLock<u64>,
    /// Number of consecutive 401 failures observed. Used to trigger manager recreation.
    consecutive_401s: AtomicU32,
}

impl TokenStore {
    /// Creates a new token store.
    ///
    /// This operation is cheap and does not actually fetch any access tokens.
    pub fn new() -> Self {
        Self {
            token: RwLock::new(None),
            token_generation: RwLock::new(0),
            consecutive_401s: AtomicU32::new(0),
        }
    }

    /// Records a successful authenticated request, resetting any failure tracking.
    pub fn record_auth_success(&self) {
        if self.consecutive_401s.load(Ordering::Relaxed) > 0 {
            self.consecutive_401s.store(0, Ordering::Relaxed);
            info!("GCS authentication recovered successfully");
        }
    }

    /// Records a 401 failure. If the failure count exceeds the threshold, triggers
    /// recreation of the `AuthenticationManager` to pick up fresh credentials.
    pub fn record_auth_failure(&self) {
        let count = self.consecutive_401s.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= MAX_CONSECUTIVE_401S_BEFORE_RESET {
            let gen = *self
                .token_generation
                .read()
                .expect("generation lock poisoned");
            reset_manager(gen);
            self.consecutive_401s.store(0, Ordering::Relaxed);
            self.invalidate();
        }
    }
}

/// Private access token module to prevent accidental leaking of tokens into logs, etc. An access
/// token can be attached to a request, but cannot be directly extracted.
mod access_token {
    use super::*;
    /// If the token data is `None`, then it is an anonymous token
    pub struct AccessToken(Option<gcp_auth::Token>);
    impl AccessToken {
        pub fn new(token: Option<gcp_auth::Token>) -> Self {
            Self(token)
        }

        /// Attaches this token to an outgoing request.
        pub fn authenticate(&self, rb: RequestBuilder) -> RequestBuilder {
            match &self.0 {
                Some(t) => rb.bearer_auth(t.as_str()),
                _ => rb,
            }
        }

        /// Tests whether this token has not expired.
        pub fn is_valid(&self) -> bool {
            match &self.0 {
                Some(t) => !t.has_expired(),
                _ => false,
            }
        }

        /// Tests whether this credential is inherently anonymous. If this returns `true`, then
        /// [`Self::authenticate`] will always return the same `RequestBuilder`.
        ///
        /// This exists as an optimization so that a [`TokenStore`] doesn't need to check locks all the
        /// time when the credential is anonymous, anyway.
        pub fn anonymous(&self) -> bool {
            !matches!(&self.0, Some(_))
        }
    }
    impl Debug for AccessToken {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }
}
use access_token::AccessToken;

impl TokenStore {
    /// Attempts to attach an access token to the given outgoing request.
    ///
    /// A cached token will be reused if it is expected to be valid for at least `lifetime`.
    /// Otherwise, a new token will be fetched and stored.
    pub fn authenticate(&self, rb: RequestBuilder) -> RequestBuilder {
        // check if access token is not none but token inside is none

        let token = self.token.read().expect("failed to read auth token");
        if let Some(t) = &*token {
            // If the token is anonymous, do nothing with the request
            if t.anonymous() {
                return rb;
            }
            // If the token is valid, authenticate the request with it
            if t.is_valid() {
                return t.authenticate(rb);
            }
        }
        drop(token);
        let mut token = self.token.write().expect("failed to write auth token");
        // Check again: may have just been written by a different client, in which case no need to
        // re-fetch.
        if let Some(t) = &*token {
            // If the token is anonymous, do nothing with the request
            if t.anonymous() {
                return rb;
            }
            // If the token is valid, authenticate the request with it
            if t.is_valid() {
                return t.authenticate(rb);
            }
        };
        // If we get here, we need a fresh token.
        match get_token() {
            Ok((t, gen)) => {
                *token = Some(t);
                *self
                    .token_generation
                    .write()
                    .expect("generation lock poisoned") = gen;
            }
            Err(e) => {
                warn!("GCS authentication failed: {}", e);
                return rb;
            }
        }
        if let Some(ref t) = *token {
            debug!("Obtained new access token.");
            // If the token is valid, authenticate the request with it
            if t.is_valid() {
                return t.authenticate(rb);
            }
        }
        rb
    }

    /// Discards any cached access token so that the next request re-fetches credentials.
    pub fn invalidate(&self) {
        let mut token = self.token.write().expect("failed to write auth token");
        *token = None;
    }
}
