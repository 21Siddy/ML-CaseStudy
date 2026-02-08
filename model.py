import numpy as np
from base_class import BaseMLModel

class Model(BaseMLModel): # This Model class fits a simple 2-layer MLP.
    def __init__(self):
        super().__init__()

        self.lr = 0.003                
        self.lr_decay = 0.99            
        self.epochs = 150               
        self.batch_size = 512           
        
        self.n_features_keep = 320      
        self.hidden1 = 256              
        self.hidden2 = 128              
        self.n_classes = 4
        
        self.noise_level = 0.04         
        self.dropout_rate = 0.15        
        self.l2_reg = 1e-5              
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        self.selected_feat_idx = None
        self.mean_ = None
        self.std_ = None
        self.weights = {}
        self.adam_m = {}
        self.adam_v = {}
        
    def _relu(self, x):
        return np.maximum(0, x)

    def _d_relu(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)

    def _init_weights(self, input_dim):
        rng = np.random.RandomState(42)
        
        w1 = rng.randn(input_dim, self.hidden1) * np.sqrt(2. / input_dim)
        b1 = np.zeros((1, self.hidden1))
        
        w2 = rng.randn(self.hidden1, self.hidden2) * np.sqrt(2. / self.hidden1)
        b2 = np.zeros((1, self.hidden2))

        w3 = rng.randn(self.hidden2, self.n_classes) * np.sqrt(1. / self.hidden2)
        b3 = np.zeros((1, self.n_classes))
        
        self.weights = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2, 'W3': w3, 'b3': b3}
        
        for k in self.weights:
            self.adam_m[k] = np.zeros_like(self.weights[k])
            self.adam_v[k] = np.zeros_like(self.weights[k])

    def _select_features(self, X):
        n_total = X.shape[1]
        
        variances = np.var(X, axis=0)
        
        top_idx = np.argsort(variances)[::-1][:self.n_features_keep]
        
        self.selected_feat_idx = np.sort(top_idx)
        print(f"  [Model] Selected {len(self.selected_feat_idx)} features (Variance only).")

    def _forward(self, X, training=False):
        W1, b1 = self.weights['W1'], self.weights['b1']
        W2, b2 = self.weights['W2'], self.weights['b2']
        W3, b3 = self.weights['W3'], self.weights['b3']
        
        z1 = X @ W1 + b1
        a1 = self._relu(z1)
        
        mask1 = None
        if training and self.dropout_rate > 0:
            mask1 = (np.random.rand(*a1.shape) > self.dropout_rate).astype(float)
            scale = 1.0 / (1.0 - self.dropout_rate)
            a1 *= mask1 * scale

        z2 = a1 @ W2 + b2
        a2 = self._relu(z2)
        
        mask2 = None
        if training and self.dropout_rate > 0:
            mask2 = (np.random.rand(*a2.shape) > self.dropout_rate).astype(float)
            scale = 1.0 / (1.0 - self.dropout_rate)
            a2 *= mask2 * scale
            
        z3 = a2 @ W3 + b3
        probs = self._softmax(z3)
        
        cache = (X, z1, a1, mask1, z2, a2, mask2, probs)
        return probs, cache

    def _backward(self, cache, y_oh, sample_weights):
        X, z1, a1, mask1, z2, a2, mask2, probs = cache
        W2 = self.weights['W2']
        W3 = self.weights['W3']
        
        N = X.shape[0]
        
        dz3 = (probs - y_oh) * sample_weights[:, None]
        
        dW3 = (a2.T @ dz3) / N + self.l2_reg * W3
        db3 = np.sum(dz3, axis=0, keepdims=True) / N
        
        da2 = dz3 @ W3.T
        if mask2 is not None:
            scale = 1.0 / (1.0 - self.dropout_rate)
            da2 *= mask2 * scale
        dz2 = da2 * self._d_relu(z2)
        
        dW2 = (a1.T @ dz2) / N + self.l2_reg * W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / N
        
        da1 = dz2 @ W2.T
        if mask1 is not None:
            scale = 1.0 / (1.0 - self.dropout_rate)
            da1 *= mask1 * scale
        dz1 = da1 * self._d_relu(z1)
        
        dW1 = (X.T @ dz1) / N + self.l2_reg * self.weights['W1']
        db1 = np.sum(dz1, axis=0, keepdims=True) / N
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        print("  [Model] Selecting features...")
        self._select_features(X)
        X = X[:, self.selected_feat_idx]
        
        # 2. Scaling
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9
        X = (X - self.mean_) / self.std_
        
        counts = np.bincount(y, minlength=self.n_classes)
        total = len(y)
        class_weights = (total / (self.n_classes * counts + 1e-5)) ** 0.6
        sample_weights = class_weights[y]
        
        print("  [Model] Oversampling minority classes...")
        X_os, y_os, w_os = [X], [y], [sample_weights]
        
        target_count = int(np.max(counts) * 0.75) 
        
        for c in range(self.n_classes):
            idx = np.where(y == c)[0]
            if len(idx) < target_count:
                n_needed = target_count - len(idx)
                dupe_idx = np.random.choice(idx, n_needed, replace=True)
                
                X_os.append(X[dupe_idx])
                y_os.append(y[dupe_idx])
                w_os.append(sample_weights[dupe_idx])
                
        X_train = np.vstack(X_os)
        y_train = np.hstack(y_os)
        w_train = np.hstack(w_os)
        
        # Shuffle
        perm = np.random.permutation(len(X_train))
        X_train, y_train, w_train = X_train[perm], y_train[perm], w_train[perm]
        
        y_oh = np.zeros((len(y_train), self.n_classes))
        y_oh[np.arange(len(y_train)), y_train] = 1
        
        self._init_weights(X_train.shape[1])
        self.t = 0
        
        print(f"  [Model] Training MLP (N={len(X_train)})...")
        
        n_batches = int(np.ceil(len(X_train) / self.batch_size))
        
        for epoch in range(self.epochs):
            self.t += 1
            epoch_loss = 0

            current_lr = self.lr * (self.lr_decay ** epoch)
            
            perm = np.random.permutation(len(X_train))
            X_curr = X_train[perm]
            y_oh_curr = y_oh[perm]
            w_curr = w_train[perm]
            
            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_curr[start:end]
                y_batch = y_oh_curr[start:end]
                w_batch = w_curr[start:end]
                
                # Noise Injection
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, X_batch.shape)
                    X_batch_noisy = X_batch + noise
                else:
                    X_batch_noisy = X_batch
                
                probs, cache = self._forward(X_batch_noisy, training=True)
                
                log_probs = -np.log(np.clip(probs, 1e-9, 1.0))
                loss = np.sum(y_batch * log_probs * w_batch[:, None]) / len(X_batch)
                epoch_loss += loss
                
                grads = self._backward(cache, y_batch, w_batch)
                self._update_params(grads, current_lr)
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/n_batches:.4f} - LR: {current_lr:.5f}")

        print("  [Model] Training complete.")
        return self

    def _update_params(self, grads, learning_rate):
        for k in self.weights:
            g = grads[k]
            self.adam_m[k] = self.beta1 * self.adam_m[k] + (1 - self.beta1) * g
            self.adam_v[k] = self.beta2 * self.adam_v[k] + (1 - self.beta2) * (g**2)
            m_hat = self.adam_m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.adam_v[k] / (1 - self.beta2 ** self.t)
            self.weights[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.selected_feat_idx is None:
            raise ValueError("Model not trained.")
        
        # 1. Prepare Data
        X = np.array(X)
        X = X[:, self.selected_feat_idx]
        X = (X - self.mean_) / self.std_
        
        preds = []
        bs = 2048 
        
        for i in range(0, len(X), bs):
            X_batch = X[i:i+bs]
            
            probs, _ = self._forward(X_batch, training=False)
            
            probs[:, 1] *= 1.10  
            probs[:, 2] *= 1.25  
            probs[:, 3] *= 1.45  
            
            preds.append(np.argmax(probs, axis=1))
            
        return np.concatenate(preds)