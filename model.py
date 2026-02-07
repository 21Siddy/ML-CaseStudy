import numpy as np
from base_class import BaseMLModel

# Implemented Machine Learning Model: Recurrent Neural Network (Elman RNN)

class Model(BaseMLModel):
    """
    A from-scratch Recurrent Neural Network (Elman RNN) for multi-class
    classification of industrial signal quality (4 classes).

    Architecture:
        1. Feature selection (variance-based + correlation filtering)
        2. Standardisation (z-score)
        3. Features are reshaped into a temporal sequence and fed through
           an Elman RNN layer.
        4. The final hidden state is projected through a fully-connected
           output layer with softmax activation.
        5. Trained with cross-entropy loss, class weighting, Adam optimiser,
           and BPTT.

    Only numpy and pandas are used — no external ML libraries.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        # --- hyper-parameters ---
        self.n_selected_features = 128     # features to keep after selection
        self.seq_len = 16                  # time steps
        self.hidden_size = 80              # RNN hidden units
        self.n_classes = 4
        self.lr = 0.0015                   # initial learning rate (for Adam)
        self.epochs = 160                  # training epochs
        self.batch_size = 128
        self.clip_value = 3.0              # gradient clipping threshold
        self.lr_decay = 0.99               # multiplicative LR decay per epoch
        self.l2_reg = 5e-5                 # L2 regularisation strength

        # Adam optimiser parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8

        # --- to be learnt / stored during fit ---
        self.selected_features_idx = None
        self.mean_ = None
        self.std_ = None

        # RNN weights (2 layers)
        self.Wx1 = None;  self.Wh1 = None;  self.bh1 = None   # layer 1
        self.Wx2 = None;  self.Wh2 = None;  self.bh2 = None   # layer 2

        # Output weights
        self.Wo = None
        self.bo = None

        # class weights for imbalanced data
        self.class_weights = None

    # ------------------------------------------------------------------
    # activation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tanh(x):
        x = np.clip(x, -10, 10)
        return np.tanh(x)

    @staticmethod
    def _tanh_deriv(tanh_out):
        return 1.0 - tanh_out ** 2

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / (np.sum(exp_vals, axis=-1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # feature selection
    # ------------------------------------------------------------------
    def _select_features(self, X):
        """
        Two-stage feature selection:
          1. Keep top features by variance.
          2. Remove highly correlated features greedily.
        """
        n_feats = X.shape[1]
        n_pre = min(self.n_selected_features * 3, n_feats)

        # --- stage 1: variance ranking ---
        variances = np.var(X, axis=0)
        top_idx = np.argsort(variances)[::-1][:n_pre]
        top_idx = np.sort(top_idx)
        X_sub = X[:, top_idx]

        # --- stage 2: correlation-based removal ---
        sample_n = min(4000, X_sub.shape[0])
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(X_sub.shape[0], sample_n, replace=False)
        X_sample = X_sub[sample_idx]

        m = X_sample.mean(axis=0)
        s = X_sample.std(axis=0)
        # mask out zero-variance features
        valid_mask = s > 1e-10
        s[~valid_mask] = 1.0
        X_std = (X_sample - m) / s
        # zero out invalid columns
        X_std[:, ~valid_mask] = 0.0

        corr = (X_std.T @ X_std) / sample_n
        # sanitise NaN/Inf
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        keep_mask = np.ones(n_pre, dtype=bool)
        # also drop zero-variance features
        keep_mask[~valid_mask] = False

        for i in range(n_pre):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_pre):
                if not keep_mask[j]:
                    continue
                if abs(corr[i, j]) > 0.85:
                    if variances[top_idx[j]] < variances[top_idx[i]]:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break

        remaining_idx = top_idx[keep_mask]
        remaining_vars = variances[remaining_idx]
        best = np.argsort(remaining_vars)[::-1][:self.n_selected_features]
        self.selected_features_idx = np.sort(remaining_idx[best])

        actual = len(self.selected_features_idx)
        # Adjust seq_len so it divides evenly
        while actual % self.seq_len != 0 and self.seq_len > 1:
            self.seq_len -= 1
        print(f"  [Model] Selected {actual} features, seq_len={self.seq_len}")

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, X, fit=False):
        X = X[:, self.selected_features_idx].astype(np.float64)

        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ < 1e-10] = 1.0

        X = (X - self.mean_) / self.std_

        N = X.shape[0]
        n_feat = len(self.selected_features_idx)
        feat_per_step = n_feat // self.seq_len
        X_seq = X[:, :self.seq_len * feat_per_step].reshape(N, self.seq_len, feat_per_step)
        return X_seq

    # ------------------------------------------------------------------
    # weight initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _orthogonal_init(rng, shape):
        """Orthogonal initialisation for recurrent weight matrices."""
        flat_shape = (shape[0], max(shape[0], shape[1]))
        a = rng.randn(*flat_shape)
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == (shape[0], shape[0]) else vt
        return q[:shape[0], :shape[1]]

    def _init_weights(self, input_dim_per_step):
        rng = np.random.RandomState(123)

        h = self.hidden_size
        d = input_dim_per_step

        # Layer 1
        limit_x = np.sqrt(2.0 / (d + h))
        self.Wx1 = rng.randn(h, d) * limit_x
        self.Wh1 = self._orthogonal_init(rng, (h, h)) * 0.5
        self.bh1 = np.zeros(h)

        # Layer 2
        limit_x2 = np.sqrt(2.0 / (h + h))
        self.Wx2 = rng.randn(h, h) * limit_x2
        self.Wh2 = self._orthogonal_init(rng, (h, h)) * 0.5
        self.bh2 = np.zeros(h)

        # Output layer
        limit_o = np.sqrt(2.0 / (h + self.n_classes))
        self.Wo = rng.randn(self.n_classes, h) * limit_o
        self.bo = np.zeros(self.n_classes)

    # ------------------------------------------------------------------
    # forward pass  (2-layer RNN)
    # ------------------------------------------------------------------
    def _forward(self, X_seq):
        """
        2-layer Elman RNN.
        Returns probs, H1_all, H2_all
        """
        batch = X_seq.shape[0]
        T = X_seq.shape[1]
        h = self.hidden_size

        H1_all = [np.zeros((batch, h))]
        H2_all = [np.zeros((batch, h))]

        for t in range(T):
            x_t = X_seq[:, t, :]
            # Layer 1
            raw1 = x_t @ self.Wx1.T + H1_all[-1] @ self.Wh1.T + self.bh1
            h1_t = self._tanh(raw1)
            H1_all.append(h1_t)

            # Layer 2
            raw2 = h1_t @ self.Wx2.T + H2_all[-1] @ self.Wh2.T + self.bh2
            h2_t = self._tanh(raw2)
            H2_all.append(h2_t)

        h_final = H2_all[-1]
        logits = h_final @ self.Wo.T + self.bo
        probs = self._softmax(logits)

        return probs, H1_all, H2_all

    # ------------------------------------------------------------------
    # backward pass  (BPTT for 2-layer RNN)
    # ------------------------------------------------------------------
    def _backward(self, X_seq, y_onehot, probs, H1_all, H2_all, sample_weights):
        batch = X_seq.shape[0]
        T = X_seq.shape[1]

        # --- output layer gradient ---
        dLogits = (probs - y_onehot) * sample_weights[:, None]

        h_final = H2_all[-1]
        dWo = dLogits.T @ h_final / batch
        dbo = dLogits.mean(axis=0)

        dh2_next = dLogits @ self.Wo  # into layer 2 final hidden

        # L2 regularisation on output weights
        dWo += self.l2_reg * self.Wo

        # --- BPTT ---
        dWx1 = np.zeros_like(self.Wx1)
        dWh1 = np.zeros_like(self.Wh1)
        dbh1 = np.zeros_like(self.bh1)

        dWx2 = np.zeros_like(self.Wx2)
        dWh2 = np.zeros_like(self.Wh2)
        dbh2 = np.zeros_like(self.bh2)

        dh1_from_above = np.zeros((batch, self.hidden_size))  # accumulated from layer 2
        dh1_next = np.zeros((batch, self.hidden_size))

        for t in reversed(range(T)):
            h2_t = H2_all[t + 1]
            h2_prev = H2_all[t]
            h1_t = H1_all[t + 1]
            h1_prev = H1_all[t]
            x_t = X_seq[:, t, :]

            # -- Layer 2 backward --
            dtanh2 = dh2_next * self._tanh_deriv(h2_t)
            dWx2 += (dtanh2.T @ h1_t) / batch
            dWh2 += (dtanh2.T @ h2_prev) / batch
            dbh2 += dtanh2.mean(axis=0)
            dh2_next = dtanh2 @ self.Wh2  # for t-1
            dh1_from_above = dtanh2 @ self.Wx2  # gradient flowing into layer 1 output

            # -- Layer 1 backward --
            dh1_total = dh1_from_above + dh1_next
            dtanh1 = dh1_total * self._tanh_deriv(h1_t)
            dWx1 += (dtanh1.T @ x_t) / batch
            dWh1 += (dtanh1.T @ h1_prev) / batch
            dbh1 += dtanh1.mean(axis=0)
            dh1_next = dtanh1 @ self.Wh1

        # L2 reg on recurrent weights
        dWx1 += self.l2_reg * self.Wx1
        dWh1 += self.l2_reg * self.Wh1
        dWx2 += self.l2_reg * self.Wx2
        dWh2 += self.l2_reg * self.Wh2

        grads = dict(Wx1=dWx1, Wh1=dWh1, bh1=dbh1,
                     Wx2=dWx2, Wh2=dWh2, bh2=dbh2,
                     Wo=dWo, bo=dbo)
        return grads

    # ------------------------------------------------------------------
    # gradient clipping (by global norm)
    # ------------------------------------------------------------------
    def _clip_grads(self, grads):
        total_norm = 0.0
        for k in grads:
            total_norm += np.sum(grads[k] ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > self.clip_value:
            scale = self.clip_value / (total_norm + 1e-12)
            for k in grads:
                grads[k] *= scale
        return grads

    # ------------------------------------------------------------------
    # Adam optimiser state
    # ------------------------------------------------------------------
    def _init_adam(self):
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0
        for name in ['Wx1', 'Wh1', 'bh1', 'Wx2', 'Wh2', 'bh2', 'Wo', 'bo']:
            param = getattr(self, name)
            self._adam_m[name] = np.zeros_like(param)
            self._adam_v[name] = np.zeros_like(param)

    def _adam_step(self, grads, lr):
        self._adam_t += 1
        t = self._adam_t
        for name in grads:
            g = grads[name]
            self._adam_m[name] = self.beta1 * self._adam_m[name] + (1 - self.beta1) * g
            self._adam_v[name] = self.beta2 * self._adam_v[name] + (1 - self.beta2) * (g ** 2)
            m_hat = self._adam_m[name] / (1 - self.beta1 ** t)
            v_hat = self._adam_v[name] / (1 - self.beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            current = getattr(self, name)
            setattr(self, name, current - update)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        """Train the 2-layer Elman RNN on (X, y)."""
        print(f"  [Model] Selecting top {self.n_selected_features} features …")
        self._select_features(X)

        print("  [Model] Preprocessing …")
        X_seq = self._preprocess(X, fit=True)
        feat_per_step = X_seq.shape[2]

        # one-hot encode targets
        y = y.astype(int)
        y_onehot = np.zeros((len(y), self.n_classes))
        y_onehot[np.arange(len(y)), y] = 1.0

        # class weights (inverse frequency, power-smoothed)
        class_counts = np.bincount(y, minlength=self.n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        w = len(y) / (self.n_classes * class_counts)
        w = w ** 0.6  # power smoothing (between linear and sqrt)
        w = np.clip(w, 0.4, 8.0)
        self.class_weights = w
        sample_w = w[y]
        print(f"  [Model] Class weights: {np.round(w, 3)}")

        # --- Oversample minority classes ---
        # Build per-class index arrays
        class_indices = [np.where(y == c)[0] for c in range(self.n_classes)]
        max_count = int(class_counts.max())
        # Oversample to match the majority class count (with noise later via shuffling)
        rng_os = np.random.RandomState(7)
        oversampled_idx = []
        for c in range(self.n_classes):
            cidx = class_indices[c]
            if len(cidx) < max_count:
                # repeat + random sample remainder
                repeats = max_count // len(cidx)
                remainder = max_count % len(cidx)
                idx = np.tile(cidx, repeats)
                idx = np.concatenate([idx, rng_os.choice(cidx, remainder, replace=False)])
            else:
                idx = cidx
            oversampled_idx.append(idx)
        oversampled_idx = np.concatenate(oversampled_idx)
        rng_os.shuffle(oversampled_idx)

        X_seq_os = X_seq[oversampled_idx]
        y_onehot_os = y_onehot[oversampled_idx]
        y_os = y[oversampled_idx]
        # For oversampled data, use uniform weights (balance already handled)
        sample_w_os = np.ones(len(y_os))

        # initialise weights + Adam state
        self._init_weights(feat_per_step)
        self._init_adam()

        N = X_seq_os.shape[0]
        rng = np.random.RandomState(0)
        lr = self.lr

        print(f"  [Model] Training 2-layer RNN  (hidden={self.hidden_size}, "
              f"seq={self.seq_len}×{feat_per_step}, epochs={self.epochs}, "
              f"oversampled N={N}) …")

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            perm = rng.permutation(N)
            X_seq_s = X_seq_os[perm]
            y_oh_s = y_onehot_os[perm]
            sw_s = sample_w_os[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                xb = X_seq_s[start:end]
                yb = y_oh_s[start:end]
                wb = sw_s[start:end]

                # forward
                probs, H1_all, H2_all = self._forward(xb)

                # NaN guard
                if np.any(np.isnan(probs)):
                    continue

                # cross-entropy loss (weighted)
                eps = 1e-12
                log_probs = -np.log(np.clip(probs, eps, 1.0))
                loss = np.sum(yb * log_probs * wb[:, None]) / len(xb)
                epoch_loss += loss

                # backward
                grads = self._backward(xb, yb, probs, H1_all, H2_all, wb)
                grads = self._clip_grads(grads)

                # Adam update
                self._adam_step(grads, lr)

                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # track best loss for early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            lr *= self.lr_decay

            if (epoch + 1) % 10 == 0 or epoch == 0:
                sub_n = min(5000, len(X_seq))
                probs_sub, _, _ = self._forward(X_seq[:sub_n])
                preds_sub = np.argmax(probs_sub, axis=1)
                acc_sub = np.mean(preds_sub == y[:sub_n])
                print(f"    epoch {epoch+1:3d}/{self.epochs}  "
                      f"loss={avg_loss:.4f}  train_acc(sub)={acc_sub:.4f}  lr={lr:.6f}")

            # early stopping if no progress for 25 epochs
            if patience_counter >= 25:
                print(f"  [Model] Early stopping at epoch {epoch+1}")
                break

        print("  [Model] Training complete.")
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0, 1, 2 or 3)."""
        X_seq = self._preprocess(X, fit=False)
        # predict in batches to avoid memory issues
        N = X_seq.shape[0]
        all_preds = []
        bs = 1024
        for start in range(0, N, bs):
            end = min(start + bs, N)
            probs, _, _ = self._forward(X_seq[start:end])
            all_preds.append(np.argmax(probs, axis=1))
        return np.concatenate(all_preds)